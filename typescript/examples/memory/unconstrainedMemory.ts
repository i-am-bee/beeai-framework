/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Message } from "beeai-framework/backend/message";

const memory = new UnconstrainedMemory();
await memory.add(
  Message.of({
    role: "user",
    text: `Hello world!`,
  }),
);

console.info(memory.isEmpty()); // false
console.log(memory.messages.length); // 1
console.log(memory.messages);
