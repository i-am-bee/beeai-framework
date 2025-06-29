/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "beeai-framework/backend/message";
import { SummarizeMemory } from "beeai-framework/memory/summarizeMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const memory = new SummarizeMemory({
  llm: new OllamaChatModel("llama3.1"),
});

await memory.addMany([
  Message.of({ role: "system", text: "You are a guide through France." }),
  Message.of({ role: "user", text: "What is the capital?" }),
  Message.of({ role: "assistant", text: "Paris" }),
  Message.of({ role: "user", text: "What language is spoken there?" }),
]);

console.info(memory.isEmpty()); // false
console.log(memory.messages.length); // 1
console.log(memory.messages[0].text); // The capital city of France is Paris, ...
