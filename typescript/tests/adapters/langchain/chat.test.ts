/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LangChainChatModel } from "@/adapters/langchain/backend/chat.js";
import { ChatModelOutput } from "@/backend/chat.js";
import { UserMessage } from "@/backend/message.js";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { AIMessageChunk } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";

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
