/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { SummarizeMemory } from "@/memory/summarizeMemory.js";
import { AssistantMessage, Message, ToolMessage, UserMessage } from "@/backend/message.js";
import { ChatModel } from "@/backend/chat.js";

function createSummarizer() {
  const prompts: string[] = [];
  const llm = {
    create: vi.fn(async ({ messages }: { messages: Message[] }) => {
      prompts.push(messages.map((msg) => msg.text).join("\n"));
      return { getTextContent: () => "SUMMARY" };
    }),
  } as unknown as ChatModel;
  return { llm, prompts };
}

describe("Summarize Memory", () => {
  it("Includes tool calls and tool results in the summary prompt", async () => {
    const { llm, prompts } = createSummarizer();
    const memory = new SummarizeMemory({ llm });

    await memory.add(new UserMessage("What's the weather in Prague?"));
    await memory.add(
      new AssistantMessage({
        type: "tool-call",
        toolCallId: "c1",
        toolName: "weather",
        input: { city: "Prague" },
      }),
    );
    await memory.add(
      new ToolMessage({
        type: "tool-result",
        toolCallId: "c1",
        toolName: "weather",
        output: { type: "text", value: "18C and raining" },
      }),
    );

    expect(prompts[1]).toContain('weather({"city":"Prague"})');
    expect(prompts[2]).toContain("18C and raining");
    expect(memory.messages[0].text).toBe("SUMMARY");
  });

  it("Keeps plain text messages unchanged", async () => {
    const { llm, prompts } = createSummarizer();
    const memory = new SummarizeMemory({ llm });

    await memory.add(new UserMessage("hello"));

    expect(prompts[0]).toContain("user: hello\n");
  });
});
