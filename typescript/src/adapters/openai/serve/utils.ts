/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatMessage } from "./types.js";
import { Message, UserMessage, AssistantMessage, SystemMessage, ToolMessage } from "@/backend/message.js";

export function openaiMessageToBeeAIMessage(msg: ChatMessage): Message {
  if (msg.role === "system") {
    return new SystemMessage(msg.content ?? "");
  } else if (msg.role === "user") {
    return new UserMessage(msg.content ?? "");
  } else if (msg.role === "tool") {
    // any cast to workaround ToolResultPart typings from different ai packages or BeeAI wrapper
    return new ToolMessage({
      type: "tool-result",
      toolCallId: msg.tool_call_id!,
      toolName: msg.name || "",
      result: msg.content,
      output: { type: "text", value: msg.content },
    } as any);
  } else if (msg.role === "assistant") {
    const assistantMsg = new AssistantMessage(msg.content ?? "");
    if (msg.tool_calls) {
      for (const call of msg.tool_calls) {
        assistantMsg.content.push({
          type: "tool-call",
          toolCallId: call.id, // Vercel AI SDK style
          toolName: call.function.name,
          args: JSON.parse(call.function.arguments || "{}"),
          input: JSON.parse(call.function.arguments || "{}"), // BeeAI style
        } as any);
      }
    }
    return assistantMsg;
  }
  throw new Error(`Unsupported role: ${msg.role}`);
}

export function transformRequestMessages(inputs: ChatMessage[]): Message[] {
  const messages: Message[] = [];
  const converted = inputs.map(openaiMessageToBeeAIMessage);

  for (let i = 0; i < converted.length; i++) {
    const msg = converted[i];
    const nextMsg = converted[i + 1];
    const nextNextMsg = converted[i + 2];

    if (msg instanceof SystemMessage) {
      continue;
    }

    // Ported from Python: Remove a handoff tool call if it's the last pair
    if (
      nextNextMsg === undefined &&
      msg instanceof AssistantMessage &&
      msg.getToolCalls().length > 0 &&
      nextMsg instanceof ToolMessage &&
      nextMsg.getToolResults().length > 0 &&
      // Vercel AI SDK uses toolCallId for both tool-call and tool-result
      (msg.getToolCalls()[0] as any).toolCallId === nextMsg.getToolResults()[0].toolCallId &&
      msg.getToolCalls()[0].toolName.toLowerCase().startsWith("transfer_to_")
    ) {
      break;
    }

    messages.push(msg);
  }

  return messages;
}
