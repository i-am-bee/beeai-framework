/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatModel, ChatModelInput, ChatModelOutput, ChatModelUsage } from "@/backend/chat.js";
import { AssistantMessage, UserMessage } from "@/backend/message.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext } from "@/context.js";
import { Serializer } from "@/serializer/serializer.js";
import { UnconstrainedCache } from "@/cache/unconstrainedCache.js";
import { NullCache } from "@/cache/nullCache.js";
import { describe, expect, it, vi } from "vitest";

class TestChatModel extends ChatModel {
  modelId = "test";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "test"], creator: this });

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  async _create(_input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
    return new ChatModelOutput([new AssistantMessage("test response")]);
  }

  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}

describe("ChatModel caching", () => {
  it("bypasses serialization when cache is disabled (NullCache default)", async () => {
    const model = new TestChatModel();
    expect(model.cache).toBeInstanceOf(NullCache);
    expect(model.cache.enabled).toBe(false);

    const serializeSpy = vi.spyOn(Serializer, "serialize");
    try {
      const response = await model.create({
        messages: [new UserMessage("Hello")],
      });
      expect(response.getTextContent()).toBe("test response");
      expect(serializeSpy).not.toHaveBeenCalled();
    } finally {
      serializeSpy.mockRestore();
    }
  });

  it("serializes when cache is enabled", async () => {
    const model = new TestChatModel();
    model.cache = new UnconstrainedCache();
    expect(model.cache.enabled).toBe(true);

    const serializeSpy = vi.spyOn(Serializer, "serialize");
    try {
      const response = await model.create({
        messages: [new UserMessage("Hello")],
      });
      expect(response.getTextContent()).toBe("test response");
      expect(serializeSpy).toHaveBeenCalled();
    } finally {
      serializeSpy.mockRestore();
    }
  });
});

describe("ChatModelOutput usage merging", () => {
  const chunk = (usage?: ChatModelUsage) =>
    new ChatModelOutput([new AssistantMessage("chunk")], usage, "stop");

  it("does not report -Infinity for optional fields the provider never set", () => {
    const usage: ChatModelUsage = {
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
      reasoningTokens: undefined,
      cachedPromptTokens: undefined,
    };

    const merged = ChatModelOutput.fromChunks([chunk(usage), chunk({ ...usage })]);

    expect(merged.usage).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
      reasoningTokens: 0,
      cachedPromptTokens: 0,
    });
    expect(Object.values(merged.usage!).every(Number.isFinite)).toBe(true);
    expect(JSON.parse(JSON.stringify(merged.usage))).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
      reasoningTokens: 0,
      cachedPromptTokens: 0,
    });
  });

  it("keeps optional fields absent when no chunk carries them", () => {
    const usage: ChatModelUsage = { promptTokens: 10, completionTokens: 5, totalTokens: 15 };

    const merged = ChatModelOutput.fromChunks([chunk(usage), chunk({ ...usage })]);

    expect(merged.usage).toEqual(usage);
    expect(merged.usage).not.toHaveProperty("reasoningTokens");
    expect(merged.usage).not.toHaveProperty("cachedPromptTokens");
  });

  it("takes the biggest value of every field across chunks", () => {
    const merged = ChatModelOutput.fromChunks([
      chunk({
        promptTokens: 10,
        completionTokens: 2,
        totalTokens: 12,
        reasoningTokens: 1,
        cachedPromptTokens: undefined,
      }),
      chunk({
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 15,
        reasoningTokens: 3,
        cachedPromptTokens: 8,
      }),
    ]);

    expect(merged.usage).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
      reasoningTokens: 3,
      cachedPromptTokens: 8,
    });
  });
});
