/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { MockLanguageModelV3 } from "ai/test";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { UserMessage } from "@/backend/message.js";
import { ChatModelError } from "@/backend/errors.js";

class TestChatModel extends VercelChatModel {}

describe("VercelChatModel", () => {
  it("does not leave an unhandled rejection when a streamed call fails", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);
    try {
      const model = new TestChatModel(
        new MockLanguageModelV3({
          doStream: async () => {
            throw new Error("simulated 400");
          },
        }),
      );
      await expect(
        model.create({ messages: [new UserMessage("hi")], stream: true }),
      ).rejects.toBeInstanceOf(ChatModelError);
      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(unhandled).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  describe("maxTokens", () => {
    const generate = async (configure: (model: TestChatModel) => void, maxTokens?: number) => {
      let maxOutputTokens: number | undefined;
      const model = new TestChatModel(
        new MockLanguageModelV3({
          doGenerate: async (options) => {
            maxOutputTokens = options.maxOutputTokens;
            return {
              content: [{ type: "text", text: "ok" }],
              finishReason: { unified: "stop", raw: undefined },
              usage: {
                inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
                outputTokens: { total: 1, text: 1, reasoning: undefined },
              },
              warnings: [],
            };
          },
        }),
      );
      configure(model);
      await model.create({ messages: [new UserMessage("hi")], maxTokens });
      return maxOutputTokens;
    };

    it("passes the per-call maxTokens to the AI SDK as maxOutputTokens", async () => {
      expect(await generate(() => {}, 42)).toBe(42);
    });

    it("passes the model-level maxTokens to the AI SDK as maxOutputTokens", async () => {
      expect(
        await generate((model) => {
          model.parameters.maxTokens = 77;
        }),
      ).toBe(77);
    });
  });
});
