/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AIMessage,
  HumanMessage as LCUserMessage,
  ToolMessage as LCToolMessage,
} from "@langchain/core/messages";
import { AssistantMessage, ToolMessage, UserMessage } from "@/backend/message.js";
import { toBeeAIMessages, toLCMessages } from "./utils.js";

describe("LangChain message utils", () => {
  it("preserves assistant tool calls when converting to LangChain messages", () => {
    const messages = toLCMessages([
      new AssistantMessage([
        { type: "text", text: "I'll check that." },
        {
          type: "tool-call",
          toolCallId: "call_1",
          toolName: "weather",
          input: { location: "Berlin" },
        },
      ]),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[0]).toMatchObject({
      content: [{ type: "text", text: "I'll check that." }],
      tool_calls: [
        {
          id: "call_1",
          name: "weather",
          args: { location: "Berlin" },
          type: "tool_call",
        },
      ],
    });
  });

  it("preserves tool results when converting to LangChain messages", () => {
    const messages = toLCMessages([
      new ToolMessage({
        type: "tool-result",
        toolCallId: "call_1",
        toolName: "weather",
        output: { type: "text", value: "Sunny" },
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(LCToolMessage);
    expect(messages[0]).toMatchObject({
      content: "Sunny",
      tool_call_id: "call_1",
      response_metadata: { tool_name: "weather" },
    });
  });

  it("preserves text and tool calls when converting from LangChain AI messages", () => {
    const messages = toBeeAIMessages([
      new AIMessage({
        content: [{ type: "text", text: "I'll check that." }],
        tool_calls: [
          {
            id: "call_1",
            name: "weather",
            args: { location: "Berlin" },
            type: "tool_call",
          },
        ],
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(AssistantMessage);
    expect(messages[0]).toMatchObject({
      role: "assistant",
      content: [
        { type: "text", text: "I'll check that." },
        {
          type: "tool-call",
          toolCallId: "call_1",
          toolName: "weather",
          input: { location: "Berlin" },
        },
      ],
    });
  });

  it("preserves tool results when converting from LangChain tool messages", () => {
    const messages = toBeeAIMessages([
      new LCToolMessage({
        content: "Sunny",
        tool_call_id: "call_1",
        response_metadata: { tool_name: "weather" },
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(ToolMessage);
    expect(messages[0]).toMatchObject({
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call_1",
          toolName: "weather",
          output: { type: "text", value: "Sunny" },
        },
      ],
    });
  });

  it("handles multiple tool results with unique IDs", () => {
    const messages = toLCMessages([
      new ToolMessage(
        [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "weather",
            output: { type: "text", value: "Sunny" },
          },
          {
            type: "tool-result",
            toolCallId: "call_2",
            toolName: "temperature",
            output: { type: "text", value: "25°C" },
          },
        ],
        undefined,
        "msg_1",
      ),
    ]);

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      id: "msg_1-0",
      content: "Sunny",
      tool_call_id: "call_1",
    });
    expect(messages[1]).toMatchObject({
      id: "msg_1-1",
      content: "25°C",
      tool_call_id: "call_2",
    });
  });

  it("handles malformed JSON in tool call args", () => {
    const messages = toLCMessages([
      new AssistantMessage([
        {
          type: "tool-call",
          toolCallId: "call_1",
          toolName: "weather",
          input: '{"location": "Berlin"',
        },
      ]),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    const aiMessage = messages[0] as AIMessage;
    expect(aiMessage.tool_calls?.[0]?.args).toEqual({ location: "Berlin" });
  });

  it("handles missing tool name with 'unknown' default", () => {
    const messages = toBeeAIMessages([
      new LCToolMessage({
        content: "Result",
        tool_call_id: "call_1",
        response_metadata: {},
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(ToolMessage);
    expect(messages[0]).toMatchObject({
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call_1",
          toolName: "unknown",
          output: { type: "text", value: "Result" },
        },
      ],
    });
  });

  it("handles image content when converting from LangChain messages", () => {
    const messages = toBeeAIMessages([
      new LCUserMessage({
        content: [
          { type: "text", text: "Check this image" },
          { type: "image", url: "https://example.com/image.jpg" },
        ],
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(UserMessage);
    expect(messages[0].content).toHaveLength(2);
    expect(messages[0].content[0]).toMatchObject({
      type: "text",
      text: "Check this image",
    });
    expect(messages[0].content[1]).toMatchObject({
      type: "image",
      image: "https://example.com/image.jpg",
    });
  });

  it("handles file content when converting from LangChain messages", () => {
    const messages = toBeeAIMessages([
      new LCUserMessage({
        content: [
          { type: "text", text: "Check this file" },
          { type: "file", file_data: "base64data", format: "application/pdf", filename: "doc.pdf" },
        ],
      }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(UserMessage);
    expect(messages[0].content).toHaveLength(2);
    expect(messages[0].content[0]).toMatchObject({
      type: "text",
      text: "Check this file",
    });
    expect(messages[0].content[1]).toMatchObject({
      type: "file",
      data: "base64data",
      mediaType: "application/pdf",
      filename: "doc.pdf",
    });
  });

  it("throws error for unknown message types", () => {
    // Create a mock unknown message by using a plain object
    const unknownMessage = {
      content: "test",
      constructor: { name: "UnknownMessage" },
    } as any;

    expect(() => {
      toBeeAIMessages([unknownMessage]);
    }).toThrow("Unsupported message type");
  });

  it("handles artifact field fallback in tool results", () => {
    const lcMessage = new LCToolMessage({
      content: "",
      tool_call_id: "call_1",
      response_metadata: { tool_name: "test" },
    });
    // Add artifact field
    (lcMessage as any).artifact = { result: "from artifact" };

    const messages = toBeeAIMessages([lcMessage]);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(ToolMessage);
    expect(messages[0].content[0]).toMatchObject({
      type: "tool-result",
      toolCallId: "call_1",
      toolName: "test",
      output: {
        type: "text",
        value: JSON.stringify({ result: "from artifact" }),
      },
    });
  });
});
