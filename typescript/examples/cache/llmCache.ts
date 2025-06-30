/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { SlidingCache } from "beeai-framework/cache/slidingCache";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { UserMessage } from "beeai-framework/backend/message";

const llm = new OllamaChatModel("llama3.1");
llm.config({
  cache: new SlidingCache({
    size: 50,
  }),
  parameters: {
    maxTokens: 25,
  },
});

console.info(await llm.cache.size()); // 0
const first = await llm.create({
  messages: [new UserMessage("Who was Alan Turing?")],
});
// upcoming requests with the EXACTLY same input will be retrieved from the cache
console.info(await llm.cache.size()); // 1
const second = await llm.create({
  messages: [new UserMessage("Who was Alan Turing?")],
});
console.info(first.getTextContent() === second.getTextContent()); // true
console.info(await llm.cache.size()); // 1
