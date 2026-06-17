/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from "vitest";
import { openaiMessageToBeeAIMessage } from "@/adapters/openai/serve/utils.js";
import { UserMessage, SystemMessage, AssistantMessage, ToolMessage } from "@/backend/message.js";

describe("OpenAIServer utils", () => {
  describe("openaiMessageToBeeAIMessage", () => {
    it("converts user messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "user",
        content: "Hello world"
      });
      expect(msg).toBeInstanceOf(UserMessage);
      expect(msg.text).toBe("Hello world");
    });

    it("converts system messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "system",
        content: "You are a helpful assistant"
      });
      expect(msg).toBeInstanceOf(SystemMessage);
      expect(msg.text).toBe("You are a helpful assistant");
    });

    it("converts assistant messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "assistant",
        content: "Sure, I can help"
      });
      expect(msg).toBeInstanceOf(AssistantMessage);
      expect(msg.text).toBe("Sure, I can help");
    });

    it("converts assistant tool calls correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_123",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "{\"location\":\"New York\"}"
            }
          }
        ]
      });
      expect(msg).toBeInstanceOf(AssistantMessage);
      const toolCalls = (msg as AssistantMessage).getToolCalls();
      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0].toolName).toBe("get_weather");
    });

    it("converts tool messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "tool",
        content: "Sunny",
        tool_call_id: "call_123",
        name: "get_weather"
      });
      expect(msg).toBeInstanceOf(ToolMessage);
      const toolResults = (msg as ToolMessage).getToolResults();
      expect(toolResults).toHaveLength(1);
      expect(toolResults[0].toolName).toBe("get_weather");
      expect(toolResults[0].toolCallId).toBe("call_123");
    });
  });
});
