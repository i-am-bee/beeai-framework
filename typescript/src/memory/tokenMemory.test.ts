/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TokenMemory } from "@/memory/tokenMemory.js";
import { AssistantMessage, Message, ToolMessage, UserMessage } from "@/backend/message.js";
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

  it("Preserves messages when synchronization rejects an oversized message", async () => {
    const instance = new TokenMemory({
      maxTokens: 4,
      syncThreshold: 1,
      capacityThreshold: 1,
      handlers: {
        estimate: () => 1,
        tokenize: async ([msg]) => (msg.text === "B" ? 5 : 2),
      },
    });
    const messages = [new UserMessage("A"), new UserMessage("B")];
    await instance.addMany(messages);

    await expect(instance.sync()).rejects.toThrow("cannot fit inside current memory");
    expect(instance.messages).toEqual(messages);
    expect(instance.stats()).toMatchObject({ tokensUsed: 2, messagesCount: 2, isDirty: true });
  });

  it("Does not keep a message rejected by the automatic sync", async () => {
    // The estimate says the message fits, but the automatic sync measures it above maxTokens.
    const instance = new TokenMemory({
      maxTokens: 6,
      syncThreshold: 0.25,
      capacityThreshold: 1,
      handlers: {
        estimate: () => 1,
        tokenize: async ([msg]) => (msg.text === "big" ? 7 : 1),
      },
    });
    const first = new UserMessage("ok");
    await instance.add(first);

    await expect(instance.add(new UserMessage("big"))).rejects.toThrow(
      "cannot fit inside current memory",
    );
    expect(instance.messages).toEqual([first]);
    expect(instance.stats()).toMatchObject({ tokensUsed: 1, messagesCount: 1 });

    const later = new UserMessage("next");
    await instance.add(later);
    expect(instance.messages).toEqual([first, later]);
    expect(instance.stats()).toMatchObject({ tokensUsed: 2, messagesCount: 2, isDirty: false });
  });

  it.each(["sync rejects the new message", "removalSelector fails after evicting"])(
    "Restores evicted messages when add fails (%s)",
    async (failure) => {
      let calls = 0;
      const instance = new TokenMemory({
        maxTokens: 2,
        syncThreshold: 2,
        capacityThreshold: 1,
        handlers: {
          estimate: (msg) =>
            msg.text === "new" && failure === "removalSelector fails after evicting" ? 2 : 1,
          tokenize: async ([msg]) => (msg.text === "new" ? 3 : 1),
          removalSelector: (msgs) => {
            calls += 1;
            if (failure === "removalSelector fails after evicting" && calls === 2) {
              throw new Error("selector failed");
            }
            return msgs[0];
          },
        },
      });
      const messages = [new UserMessage("a"), new UserMessage("b")];
      await instance.addMany(messages);
      await instance.sync();
      (instance as unknown as { syncThreshold: number }).syncThreshold = 0.25;

      await expect(instance.add(new UserMessage("new"))).rejects.toThrow();
      expect(instance.messages).toEqual(messages);
      expect(instance.stats()).toMatchObject({ tokensUsed: 2, messagesCount: 2, isDirty: false });
    },
  );

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

  const toolExchange = (id: string, result: string) => [
    new AssistantMessage({
      type: "tool-call",
      toolCallId: id,
      toolName: "fetch",
      input: { url: "https://example.com" },
    }),
    new ToolMessage({
      type: "tool-result",
      toolCallId: id,
      toolName: "fetch",
      output: { type: "text", value: result },
    }),
  ];

  it("Counts tool calls and results against the budget", async () => {
    const instance = new TokenMemory({ maxTokens: 2000 });
    for (let i = 0; i < 5; i++) {
      await instance.addMany([
        new UserMessage(`fetch page ${i}`),
        ...toolExchange(`call_${i}`, "x".repeat(4000)),
      ]);
    }

    // Each result is ~1000 tokens, so the 1500-token capacity holds one exchange, not all five.
    expect(instance.tokensUsed).toBeLessThanOrEqual(2000 * 0.75);
    expect(instance.messages.filter((msg) => msg instanceof ToolMessage)).toHaveLength(1);
  });

  it("Evicts a tool call together with its result", async () => {
    const instance = new TokenMemory({ maxTokens: 200 });
    await instance.addMany(toolExchange("call_1", "x".repeat(200)));

    // Evicting the call alone (~11 tokens) would make room, but the result must not stay behind without it.
    const question = new UserMessage("y".repeat(340));
    await instance.add(question);

    expect(instance.messages).toEqual([question]);
  });

  const toolResult = (id: string, result: string) =>
    new ToolMessage({
      type: "tool-result",
      toolCallId: id,
      toolName: "fetch",
      output: { type: "text", value: result },
    });

  // syncThreshold 2 never syncs, so add() alone decides; the default syncs after each add and replays eviction.
  it.each([
    ["add", 2],
    ["sync", 0.25],
  ])("Keeps the call of an oversized tool result (%s)", async (_, syncThreshold) => {
    const instance = new TokenMemory({ maxTokens: 2000, syncThreshold });
    const [call] = toolExchange("call_1", "");
    const result = toolResult("call_1", "x".repeat(7200));
    await instance.addMany([new UserMessage("fetch the page"), call!]);

    // The ~1800-token result exceeds the 1500-token capacity alone; its call stays even though that is over budget.
    await instance.add(result);

    expect(instance.messages).toEqual([call, result]);
  });

  it.each([
    ["add", 2],
    ["sync", 0.25],
  ])(
    "Keeps the call and first result when a second result arrives (%s)",
    async (_, syncThreshold) => {
      const instance = new TokenMemory({ maxTokens: 2000, syncThreshold });
      const call = new AssistantMessage([
        {
          type: "tool-call",
          toolCallId: "call_1",
          toolName: "fetch",
          input: { url: "https://example.com/a" },
        },
        {
          type: "tool-call",
          toolCallId: "call_2",
          toolName: "fetch",
          input: { url: "https://example.com/b" },
        },
      ]);
      const first = toolResult("call_1", "x".repeat(2800));
      const second = toolResult("call_2", "x".repeat(3600));
      await instance.addMany([new UserMessage("fetch both pages"), call, first]);

      await instance.add(second);

      expect(instance.messages).toEqual([call, first, second]);

      // The next message outside the exchange evicts the whole exchange.
      const reply = new AssistantMessage("done");
      await instance.add(reply);
      expect(instance.messages).toEqual([reply]);
    },
  );

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
