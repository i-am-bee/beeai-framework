/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { Message } from "@/backend/message.js";

describe("Unconstrained Memory", () => {
  const date = new Date();
  const createMessage = (i: number) =>
    Message.of({ role: "user", text: `${i}`, meta: { createdAt: new Date(date) } });

  it("Splices", async () => {
    const memory = new UnconstrainedMemory();
    const messages = Array(10)
      .fill(0)
      .map((_, i) => createMessage(i + 1));
    await memory.addMany(messages);

    expect(memory.messages).toStrictEqual(messages);
    await memory.splice(-1, 1, createMessage(10));
    expect(memory.messages).toStrictEqual(messages);
    await memory.splice(-1, 2, createMessage(10));
    expect(memory.messages).toStrictEqual(messages);
    await memory.splice(-2, 2, createMessage(9), createMessage(10));
    expect(memory.messages).toStrictEqual(messages);
    await memory.splice(0, 2, createMessage(1), createMessage(2));
    expect(memory.messages).toStrictEqual(messages);
  });
});
