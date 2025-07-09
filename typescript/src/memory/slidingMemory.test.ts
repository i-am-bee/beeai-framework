/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { SlidingMemory } from "@/memory/slidingMemory.js";
import { Message, UserMessage } from "@/backend/message.js";

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
    const deserialized = await SlidingMemory.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
