/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TokenMemory } from "@/memory/tokenMemory.js";
import { Message, UserMessage } from "@/backend/message.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";
import { sum } from "remeda";

describe("Token Memory", () => {
  const getInstance = (config: {
    llmFactor: number;
    localFactor: number;
    syncThreshold: number;
    maxTokens: number;
    capacityThreshold?: number;
  }) => {
    return new TokenMemory({
      maxTokens: config.maxTokens,
      syncThreshold: config.syncThreshold,
      // 1 isolates a case to pure maxTokens enforcement
      capacityThreshold: config.capacityThreshold ?? 1,
      handlers: {
        estimate: (msg) => Math.ceil(msg.text.length * config.localFactor),
        tokenize: async (msgs) =>
          sum(msgs.map((msg) => Math.ceil(msg.text.length * config.llmFactor))),
      },
    });
  };

  it("Auto sync", async () => {
    const instance = getInstance({
      llmFactor: 2,
      localFactor: 1,
      maxTokens: 4,
      syncThreshold: 0.5,
    });
    await instance.addMany([
      new UserMessage("A"),
      new UserMessage("B"),
      new UserMessage("C"),
      new UserMessage("D"),
    ]);
    expect(instance.stats()).toMatchObject({
      isDirty: false,
      tokensUsed: 4,
      messagesCount: 2,
    });
  });

  it("Synchronizes", async () => {
    const instance = getInstance({
      llmFactor: 2,
      localFactor: 1,
      maxTokens: 10,
      syncThreshold: 1,
    });
    expect(instance.stats()).toMatchObject({
      isDirty: false,
      tokensUsed: 0,
      messagesCount: 0,
    });
    await instance.addMany([
      new UserMessage("A"),
      new UserMessage("B"),
      new UserMessage("C"),
      new UserMessage("D"),
      new UserMessage("E"),
      new UserMessage("F"),
    ]);
    expect(instance.stats()).toMatchObject({
      isDirty: true,
      tokensUsed: 6,
      messagesCount: 6,
    });
    await instance.sync();
    expect(instance.stats()).toMatchObject({
      isDirty: false,
      tokensUsed: 10,
      messagesCount: 5,
    });
  });

  it("Evicts down to the capacity threshold", async () => {
    const instance = getInstance({
      llmFactor: 1,
      localFactor: 1,
      maxTokens: 6,
      syncThreshold: 1,
      capacityThreshold: 0.5,
    });

    // The budget is 6 * 0.5 = 3 tokens, so only the most recent 3-token message fits
    // even though maxTokens on its own would allow both.
    await instance.add(new UserMessage("aaa"));
    await instance.add(new UserMessage("bbb"));

    expect(instance.messages.map((msg) => msg.text)).toEqual(["bbb"]);
    expect(instance.tokensUsed).toBeLessThanOrEqual(3);
  });

  it("Rejects a capacityThreshold above 1", () => {
    expect(() => new TokenMemory({ capacityThreshold: 1.5 })).toThrowError(TypeError);
  });

  it("Serializes", async () => {
    const instance = getInstance({
      llmFactor: 2,
      localFactor: 1,
      maxTokens: 10,
      syncThreshold: 1,
    });
    await instance.add(
      Message.of({
        text: "Hello!",
        role: "user",
      }),
    );
    const serialized = await instance.serialize();
    const deserialized = await TokenMemory.fromSerialized(serialized, {
      allowFunctionDeserialization: true,
    });
    verifyDeserialization(instance, deserialized);
  });
});
