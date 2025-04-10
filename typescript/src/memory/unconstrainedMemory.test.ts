/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
