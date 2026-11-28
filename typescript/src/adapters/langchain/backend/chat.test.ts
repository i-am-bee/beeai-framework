/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LangChainChatModel } from "@/adapters/langchain/backend/chat.js";
import { ChatModelOutput } from "@/backend/chat.js";
import { UserMessage } from "@/backend/message.js";
import { DynamicTool, StringToolOutput } from "@/tools/base.js";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { AIMessageChunk } from "@langchain/core/messages";
import { describe, expect, it, onTestFinished, vi } from "vitest";
import { z } from "zod";

const finishReasonFor = (response_metadata: Record<string, unknown>) => {
  const model = new LangChainChatModel(new FakeListChatModel({ responses: ["hi"] }));
  const prepare = (
    model as unknown as {
      prepareOutput: (o: AIMessageChunk) => { finishReason?: string };
    }
  ).prepareOutput.bind(model);
  return prepare(new AIMessageChunk({ content: "hi", response_metadata })).finishReason;
};

describe("LangChain chat adapter finish reason", () => {
  it.each([
    [{ finish_reason: "stop" }, "stop"],
    [{ finish_reason: "length" }, "length"],
    [{ finish_reason: "tool_calls" }, "tool-calls"],
    [{ finish_reason: "content_filter" }, "content-filter"],
    [{ stop_reason: "end_turn" }, "stop"],
    [{ stop_reason: "max_tokens" }, "length"],
    [{ stop_reason: "tool_use" }, "tool-calls"],
    [{ stop_reason: "refusal" }, "content-filter"],
    [{ done_reason: "length" }, "length"],
    [{ stopReason: "tool_use" }, "tool-calls"],
    [{ messageStop: { stopReason: "max_tokens" } }, "length"],
    [{ finishReason: "MAX_TOKENS" }, "length"],
    [{ finish_reason: "something_new" }, "other"],
    [{}, undefined],
  ])("maps %j to %s", (meta, expected) => {
    expect(finishReasonFor(meta)).toBe(expected);
  });

  it("creates a model output", async () => {
    const model = new LangChainChatModel(new FakeListChatModel({ responses: ["hi"] }));
    const out = await model.create({ messages: [new UserMessage("x")] });
    expect(out.getTextContent()).toBe("hi");
  });

  it("keeps the last finish reason when a trailing chunk has none", () => {
    const model = new LangChainChatModel(new FakeListChatModel({ responses: ["hi"] }));
    const prepare = (
      model as unknown as { prepareOutput: (o: AIMessageChunk) => ChatModelOutput }
    ).prepareOutput.bind(model);
    const chunks = [
      prepare(new AIMessageChunk({ content: "a", response_metadata: {} })),
      prepare(new AIMessageChunk({ content: "b", response_metadata: { finish_reason: "length" } })),
      prepare(new AIMessageChunk({ content: "", response_metadata: {} })),
    ];
    expect(ChatModelOutput.fromChunks(chunks).finishReason).toBe("length");
  });
});

describe("LangChain chat adapter tools", () => {
  const weather = new DynamicTool({
    name: "weather",
    description: "Current weather",
    inputSchema: z.object({ city: z.string() }),
    handler: async () => new StringToolOutput("sunny"),
  });

  it.each([
    [weather, "weather", '{"city":"Paris"}'],
    ["auto", "auto", "hi"],
  ] as const)("sends tools and tool choice %#", async (toolChoice, expected, response) => {
    const llm = new FakeListChatModel({ responses: [response] });
    const bindTools = vi.fn((_tools: unknown[]) => llm);
    Object.assign(llm, { bindTools });
    const generate = vi.spyOn(FakeListChatModel.prototype, "_generate");
    onTestFinished(() => generate.mockRestore());

    await new LangChainChatModel(llm).create({
      messages: [new UserMessage("x")],
      tools: [weather],
      toolChoice,
    });

    expect(bindTools.mock.calls[0][0]).toEqual([
      {
        type: "function",
        function: {
          name: "weather",
          description: "Current weather",
          parameters: await weather.getInputJsonSchema(),
        },
      },
    ]);
    expect(generate.mock.calls[0][1]).toHaveProperty("tool_choice", expected);
  });
});
