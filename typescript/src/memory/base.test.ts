/**
 * Copyright 2026 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "@/backend/message.js";
import { SlidingMemory } from "@/memory/slidingMemory.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";

describe.each([
  { name: "UnconstrainedMemory", create: () => new UnconstrainedMemory() },
  { name: "SlidingMemory", create: () => new SlidingMemory({ size: 5 }) },
])("$name deleteMany", ({ create }) => {
  const message = (text: string) => Message.of({ role: "user", text });

  it("deletes every message when given the live messages array", async () => {
    const memory = create();
    await memory.addMany([message("first"), message("second"), message("third")]);

    await memory.deleteMany(memory.messages);

    expect(memory.messages).toEqual([]);
  });

  it("deletes adjacent matches from a filtered live iterator", async () => {
    const memory = create();
    const expired = new Set([message("expired first"), message("expired second")]);
    const retained = message("retained");
    await memory.addMany([...expired, retained]);

    function* selection() {
      for (const candidate of memory) {
        if (expired.has(candidate)) {
          yield candidate;
        }
      }
    }

    await memory.deleteMany(selection());

    expect(memory.messages).toEqual([retained]);
  });

  it("deletes adjacent matches from an async live iterator", async () => {
    const memory = create();
    const expired = new Set([message("expired first"), message("expired second")]);
    const retained = message("retained");
    await memory.addMany([...expired, retained]);

    async function* selection() {
      for (const candidate of memory) {
        await Promise.resolve();
        if (expired.has(candidate)) {
          yield candidate;
        }
      }
    }

    await memory.deleteMany(selection());

    expect(memory.messages).toEqual([retained]);
  });

  it("preserves unselected messages when given a detached selection", async () => {
    const memory = create();
    const first = message("first");
    const second = message("second");
    const retained = message("retained");
    await memory.addMany([first, second, retained]);

    await memory.deleteMany([first, second]);

    expect(memory.messages).toEqual([retained]);
  });

  it("leaves history unchanged for an empty selection", async () => {
    const memory = create();
    const retained = [message("first"), message("second")];
    await memory.addMany(retained);

    await memory.deleteMany([]);

    expect(memory.messages).toEqual(retained);
  });
});
