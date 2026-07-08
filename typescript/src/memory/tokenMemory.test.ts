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
  }) => {
    return new TokenMemory({
      maxTokens: config.maxTokens,
      syncThreshold: config.syncThreshold,
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
    const deserialized = await TokenMemory.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
