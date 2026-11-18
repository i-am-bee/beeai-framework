/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { SlidingMemory } from "@/memory/slidingMemory.js";
import { AssistantMessage, Message, ToolMessage, UserMessage } from "@/backend/message.js";
import { MemoryFatalError } from "@/memory/base.js";

describe("Sliding Memory", () => {
  it("Removes old messages", async () => {
    const instance = new SlidingMemory({
      size: 2,
    });
    await instance.addMany(["A", "B", "C"].map((text) => Message.of({ role: "user", text })));
    expect(Array.from(instance).map((msg) => msg.text)).toStrictEqual(["B", "C"]);
  });

  it("Removes messages by removalSelector", async () => {
    const instance = new SlidingMemory({
      size: 2,
      handlers: {
        removalSelector: (messages) => messages.find((msg) => msg.role !== "system")!,
      },
    });
    await instance.add(Message.of({ role: "system", text: "You are a helpful assistant." }));
    await instance.addMany(["A", "B", "C"].map((text) => new UserMessage(text)));
    expect(Array.from(instance).map((msg) => msg.text)).toStrictEqual([
      "You are a helpful assistant.",
      "C",
    ]);
  });

  it("Removes multiple messages by removalSelector", async () => {
    const instance = new SlidingMemory({
      size: 5,
      handlers: {
        removalSelector: (messages) => {
          const index = messages.findIndex((msg, index, arr) => {
            const next = arr[index + 1];
            return (
              (msg.role === "user" && next?.role === "assistant") ||
              (msg.role === "assistant" && next?.role === "user")
            );
          });
          if (index === -1) {
            return messages.find((msg) => msg.role === "user")!;
          }
          return [messages[index], messages[index + 1]];
        },
      },
    });
    const roles = ["user", "assistant", "user", "assistant", "user", "user"] as const;
    await instance.addMany(roles.map((role, i) => Message.of({ role, text: `${i + 1}` })));
    expect(Array.from(instance).map((msg) => msg.text)).toStrictEqual(["3", "4", "5", "6"]);
  });

  it("Serializes", async () => {
    const instance = new SlidingMemory({
      size: 5,
    });
    await instance.add(new UserMessage("Hello!"));
    const serialized = await instance.serialize();
    const deserialized = await SlidingMemory.fromSerialized(serialized, {
      allowFunctionDeserialization: true,
    });
    verifyDeserialization(instance, deserialized);
  });

  const call = (...ids: string[]) =>
    new AssistantMessage(
      ids.map((id) => ({
        type: "tool-call" as const,
        toolName: "weather",
        toolCallId: id,
        input: {},
      })),
    );
  const result = (id: string) =>
    new ToolMessage({
      type: "tool-result",
      toolName: "weather",
      toolCallId: id,
      output: { type: "text", value: "sunny" },
    });

  it("Removes an assistant call together with all its tool results", async () => {
    const memory = new SlidingMemory({ size: 4 });
    const question = new UserMessage("next question");
    const answer = new AssistantMessage("next answer");
    await memory.addMany([call("a", "b"), result("a"), result("b"), question]);
    await memory.add(answer);
    expect(memory.messages).toEqual([question, answer]);
  });

  it("Removes the complete exchange when a selector chooses a tool result", async () => {
    const memory = new SlidingMemory({
      size: 4,
      handlers: { removalSelector: (messages) => messages[2] },
    });
    const first = new UserMessage("first");
    const next = new UserMessage("next");
    const last = new UserMessage("last");
    await memory.addMany([first, call("a"), result("a"), next]);
    await memory.add(last);
    expect(memory.messages).toEqual([first, next, last]);
  });

  it("Accepts selectors that include several members of the same exchange", async () => {
    const memory = new SlidingMemory({
      size: 3,
      handlers: { removalSelector: (messages) => messages.slice(0, 2) },
    });
    const next = new UserMessage("next");
    await memory.addMany([call("a", "b"), result("a"), result("b")]);
    await memory.add(next);
    expect(memory.messages).toEqual([next]);
  });

  it("Makes room for a new result without evicting its call", async () => {
    const memory = new SlidingMemory({ size: 2 });
    const pending = call("a");
    const response = result("a");
    await memory.addMany([new UserMessage("question"), pending]);
    await memory.add(response);
    expect(memory.messages).toEqual([pending, response]);
  });

  it("Leaves an exchange unchanged when the window cannot fit another result", async () => {
    const memory = new SlidingMemory({ size: 2 });
    const pending = call("a", "b");
    const first = result("a");
    await memory.addMany([pending, first]);
    await expect(memory.add(result("b"))).rejects.toBeInstanceOf(MemoryFatalError);
    expect(memory.messages).toEqual([pending, first]);
  });

  it("Validates the full grouped eviction before changing history", async () => {
    const memory = new SlidingMemory({
      size: 3,
      handlers: { removalSelector: (messages) => [messages[0], new UserMessage("not stored")] },
    });
    const stored = [call("a"), result("a"), new UserMessage("next")];
    await memory.addMany(stored);
    await expect(memory.add(new UserMessage("last"))).rejects.toBeInstanceOf(MemoryFatalError);
    expect(memory.messages).toEqual(stored);
  });

  it("Removes consecutive assistant calls with their results", async () => {
    const memory = new SlidingMemory({ size: 5 });
    const question = new UserMessage("next question");
    const answer = new AssistantMessage("next answer");
    await memory.addMany([call("a"), call("b"), result("a"), result("b"), question]);
    await memory.add(answer);
    expect(memory.messages).toEqual([question, answer]);
  });

  it("Rejects duplicate selections without losing the exchange", async () => {
    const memory = new SlidingMemory({
      size: 3,
      handlers: { removalSelector: (messages) => [messages[0], messages[1], messages[1]] },
    });
    const stored = [call("a"), result("a"), new UserMessage("next")];
    await memory.addMany(stored);
    await expect(memory.add(new UserMessage("last"))).rejects.toBeInstanceOf(MemoryFatalError);
    expect(memory.messages).toEqual(stored);
  });

  it("Rejects an empty selection without changing history", async () => {
    const memory = new SlidingMemory({ size: 2, handlers: { removalSelector: () => [] } });
    const stored = [call("a"), result("a")];
    await memory.addMany(stored);
    await expect(memory.add(new UserMessage("next"))).rejects.toBeInstanceOf(MemoryFatalError);
    expect(memory.messages).toEqual(stored);
  });
});
