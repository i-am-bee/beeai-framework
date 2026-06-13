/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AIMessage as LCAIMessage,
  AIMessageChunk,
  BaseMessage as LCBaseMessage,
  HumanMessage as LCUserMessage,
  SystemMessage as LCSystemMessage,
  ToolMessage as LCToolMessage,
} from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import {
  AssistantMessage,
  Message,
  SystemMessage,
  ToolMessage,
  UserMessage,
} from "@/backend/message.js";
import { FilePart, ImagePart, TextPart, ToolCallPart, ToolResultPart } from "ai";
import { isPlainObject, isString } from "remeda";
import { parseBrokenJson } from "@/internals/helpers/schema.js";
import { ValueError } from "@/errors.js";
import { Logger } from "@/logger/logger.js";

/**
 * Removes falsy values from an array.
 * @param arr - Array to filter
 * @returns Array with falsy values removed
 */
function removeFalsy<T>(arr: (T | null | undefined | false | 0 | "")[]): T[] {
  return arr.filter((item): item is T => Boolean(item));
}

/**
 * Converts a BeeAI Message to LangChain content format.
 * Supports text, image, and file content types.
 * @param message - The BeeAI message to convert
 * @returns String content or array of content parts
 */
function toLCContent(
  message: Message,
): string | { type: string; text?: string; url?: string; [key: string]: any }[] {
  if (message instanceof ToolMessage) {
    const toolResult = message.getToolResults()[0];
    if (!toolResult) {
      Logger.root.warn("ToolMessage has no tool results, returning empty string");
      return "";
    }

    if ("value" in toolResult.output && isString(toolResult.output.value)) {
      return toolResult.output.value;
    }

    return JSON.stringify(toolResult.output);
  }

  const lcContent: ({ type: string; text?: string; url?: string; [key: string]: any } | null)[] =
    [];

  for (const part of message.content) {
    if (part.type === "text") {
      lcContent.push({ type: "text", text: part.text });
    } else if (part.type === "image") {
      const imageUrl = part.image instanceof URL ? part.image.toString() : String(part.image);
      lcContent.push({ type: "image", url: imageUrl });
    } else if (part.type === "file") {
      lcContent.push({
        type: "file",
        file_data: part.data,
        format: part.mediaType,
        ...(part.filename && { filename: part.filename }),
      });
    } else {
      lcContent.push(null);
    }
  }

  return removeFalsy(lcContent);
}

/**
 * Converts tool call input to LangChain arguments format.
 * Parses JSON strings and handles broken JSON gracefully.
 * @param input - The tool call input (object or JSON string)
 * @returns Tool arguments as a record object
 */
function toLCToolArgs(input: unknown): Record<string, any> {
  if (isPlainObject(input)) {
    return input;
  }

  if (isString(input)) {
    const parsed = parseBrokenJson(input, { pair: ["{", "}"] });
    if (isPlainObject(parsed)) {
      return parsed;
    }
  }

  return {};
}

/**
 * Converts AssistantMessage tool calls to LangChain format.
 * @param message - The assistant message with tool calls
 * @returns Array of LangChain tool call objects
 */
function toLCToolCalls(message: AssistantMessage) {
  return message.getToolCalls().map((toolCall) => ({
    id: toolCall.toolCallId,
    name: toolCall.toolName,
    args: toLCToolArgs(toolCall.input),
    type: "tool_call" as const,
  }));
}

/**
 * Converts LangChain message content to BeeAI TextPart array.
 * Handles strings, arrays, and text objects. Filters out empty values.
 * @param content - The LangChain message content
 * @returns Array of BeeAI TextPart objects
 */
function toBeeAITextParts(content: unknown): TextPart[] {
  if (typeof content === "string") {
    return content ? [{ type: "text", text: content }] : [];
  }

  if (!Array.isArray(content)) {
    if (content !== null && content !== undefined) {
      Logger.root.warn(`Unexpected content type in toBeeAITextParts: ${typeof content}`);
    }
    return [];
  }

  return content.flatMap((part) => {
    if (typeof part === "string") {
      return part ? [{ type: "text", text: part }] : [];
    }

    if (
      part &&
      typeof part === "object" &&
      "type" in part &&
      part.type === "text" &&
      "text" in part
    ) {
      return [{ type: "text", text: String(part.text ?? "") }];
    }

    return [];
  });
}

/**
 * Converts LangChain message content to BeeAI content parts for UserMessage.
 * Supports text, image (URL or base64), and file content types.
 * @param content - The LangChain message content
 * @returns Array of TextPart, ImagePart, or FilePart objects
 */
function toBeeAIUserContentParts(content: unknown): (TextPart | ImagePart | FilePart)[] {
  if (typeof content === "string") {
    return content ? [{ type: "text", text: content }] : [];
  }

  if (!Array.isArray(content)) {
    if (content !== null && content !== undefined) {
      Logger.root.warn(`Unexpected content type in toBeeAIContentParts: ${typeof content}`);
    }
    return [];
  }

  const results: (TextPart | ImagePart | FilePart | null)[] = [];

  for (const part of content) {
    if (typeof part === "string") {
      if (part) {
        results.push({ type: "text", text: part });
      }
      continue;
    }

    if (!part || typeof part !== "object" || !("type" in part)) {
      continue;
    }

    if (part.type === "text" && "text" in part) {
      results.push({ type: "text", text: String(part.text ?? "") });
      continue;
    }

    if (part.type === "image") {
      let imageUrl = "";

      if ("url" in part && part.url) {
        imageUrl = String(part.url);
      } else if ("data" in part && "mime_type" in part) {
        imageUrl = `data:${part.mime_type};base64,${part.data}`;
      }

      if (imageUrl) {
        results.push({
          type: "image",
          image: imageUrl,
        });
      }
      continue;
    }

    if (part.type === "file") {
      let fileData = "";
      let mediaType = "application/octet-stream";
      let filename: string | undefined;

      if ("file_id" in part && part.file_id) {
        fileData = String(part.file_id);
      } else if ("file_data" in part && part.file_data) {
        fileData = String(part.file_data);
      }

      if ("format" in part && part.format) {
        mediaType = String(part.format);
      } else if ("mime_type" in part && part.mime_type) {
        mediaType = String(part.mime_type);
      }

      if ("filename" in part && part.filename) {
        filename = String(part.filename);
      }

      if (fileData) {
        results.push({
          type: "file",
          data: fileData,
          mediaType,
          ...(filename && { filename }),
        });
      }
    }
  }

  return removeFalsy(results);
}

/**
 * Converts a LangChain tool call to BeeAI ToolCallPart format.
 * @param toolCall - The LangChain tool call object
 * @returns BeeAI ToolCallPart
 */
function toBeeAIToolCall(toolCall: { id?: string; name?: string; args?: unknown }): ToolCallPart {
  return {
    type: "tool-call",
    toolCallId: toolCall.id ?? "",
    toolName: toolCall.name ?? "",
    input: toolCall.args ?? {},
  };
}

/**
 * Converts a LangChain ToolMessage to BeeAI ToolResultPart.
 * Falls back to artifact field if content is empty.
 * @param message - The LangChain tool message
 * @returns BeeAI ToolResultPart object
 */
function toBeeAIToolResult(message: LCToolMessage): ToolResultPart {
  let content: string;
  if (typeof message.content === "string" && message.content) {
    content = message.content;
  } else if (message.content) {
    content = JSON.stringify(message.content);
  } else if ("artifact" in message && message.artifact) {
    content = JSON.stringify(message.artifact);
  } else {
    content = "";
  }

  return {
    type: "tool-result",
    toolCallId: message.tool_call_id,
    toolName: String(message.response_metadata?.tool_name ?? "unknown"),
    output: {
      type: "text",
      value: content,
    },
  };
}

/**
 * Converts BeeAI Messages to LangChain BaseMessage format.
 * Handles UserMessage, AssistantMessage, ToolMessage, and SystemMessage.
 * ToolMessages with multiple results are split into separate messages with ID suffixes.
 * Unknown types fallback to LCUserMessage.
 * @param messages - Array of BeeAI messages
 * @returns Array of LangChain BaseMessage objects
 */
export function toLCMessages(messages: Message[]): LCBaseMessage[] {
  const output: LCBaseMessage[] = [];

  for (const msg of messages) {
    const content = toLCContent(msg);

    if (msg instanceof UserMessage) {
      output.push(new LCUserMessage({ content, id: msg.id }));
      continue;
    }

    if (msg instanceof AssistantMessage) {
      output.push(
        new LCAIMessage({
          content,
          id: msg.id,
          tool_calls: toLCToolCalls(msg),
        }),
      );
      continue;
    }

    if (msg instanceof ToolMessage) {
      const toolResults = msg.getToolResults();
      for (let i = 0; i < toolResults.length; i++) {
        const toolResult = toolResults[i];
        const messageId = toolResults.length > 1 ? `${msg.id}-${i}` : msg.id;
        output.push(
          new LCToolMessage({
            id: messageId,
            content:
              "value" in toolResult.output && isString(toolResult.output.value)
                ? toolResult.output.value
                : JSON.stringify(toolResult.output),
            tool_call_id: toolResult.toolCallId,
            response_metadata: { tool_name: toolResult.toolName },
          }),
        );
      }
      continue;
    }

    if (msg instanceof SystemMessage) {
      output.push(new LCSystemMessage({ content, id: msg.id }));
      continue;
    }

    output.push(new LCUserMessage({ content, id: msg.id }));
  }

  return output;
}

/**
 * Converts LangChain BaseMessages to BeeAI Message format.
 * Handles LCUserMessage, LCAIMessage, AIMessageChunk, LCSystemMessage, and LCToolMessage.
 * Throws ValueError for unknown message types.
 * @param messages - Array of LangChain messages
 * @returns Array of BeeAI Message objects
 */
export function toBeeAIMessages(messages: LCBaseMessage[]): Message[] {
  return messages.map((msg) => {
    if (msg instanceof LCUserMessage) {
      return new UserMessage(toBeeAIUserContentParts(msg.content), msg.response_metadata, msg.id);
    }

    if (msg instanceof LCAIMessage || msg instanceof AIMessageChunk) {
      return new AssistantMessage(
        [...toBeeAITextParts(msg.content), ...(msg.tool_calls ?? []).map(toBeeAIToolCall)],
        msg.response_metadata,
        msg.id,
      );
    }

    if (msg instanceof LCSystemMessage) {
      return new SystemMessage(toBeeAITextParts(msg.content), msg.response_metadata, msg.id);
    }

    if (msg instanceof LCToolMessage) {
      return new ToolMessage(toBeeAIToolResult(msg), msg.response_metadata, msg.id);
    }

    throw new ValueError(`Unsupported message type: ${msg.constructor.name}`);
  });
}

/**
 * Converts a BeeAI Tool to LangChain DynamicStructuredTool.
 * Wraps the tool's run method and extracts text content from results.
 * @param tool - The BeeAI tool to convert
 * @returns Promise resolving to LangChain DynamicStructuredTool
 */
export async function beeaiToolToLCTool(tool: any): Promise<DynamicStructuredTool> {
  const schema = await tool.inputSchema();

  return new DynamicStructuredTool({
    name: tool.name,
    description: tool.description,
    schema,
    func: async (input: any) => {
      const result = await tool.run(input);
      return result.getTextContent();
    },
  });
}
