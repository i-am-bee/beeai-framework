/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import "dotenv/config.js";
import { createConsoleReader } from "examples/helpers/io.js";
import { UserMessage } from "beeai-framework/backend/message";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const llm = new OllamaChatModel("llama3.1");

//  Optionally one may set llm parameters
llm.parameters.maxTokens = 10000; // high number yields longer potential output
llm.parameters.topP = 0; // higher number yields more complex vocabulary, recommend only changing p or k
llm.parameters.frequencyPenalty = 0; // higher number yields reduction in word reptition
llm.parameters.temperature = 0; // higher number yields greater randomness and variation
llm.parameters.topK = 0; // higher number yields more variance, recommend only changing p or k
llm.parameters.n = 1; // higher number yields more choices
llm.parameters.presencePenalty = 0; // higher number yields reduction in repetition of words
llm.parameters.seed = 10; // can help produce similar responses if prompt and seed are always the same
llm.parameters.stopSequences = ["q", "quit", "ahhhhhhhhh"]; // stops the model on input of any of these strings

// alternatively
llm.config({
  parameters: {
    maxTokens: 10000,
    // other parameters
  },
});

const reader = createConsoleReader();

for await (const { prompt } of reader) {
  const response = await llm.create({
    messages: [new UserMessage(prompt)],
  });
  reader.write(`LLM 🤖 (txt) : `, response.getTextContent());
  reader.write(`LLM 🤖 (raw) : `, JSON.stringify(response.messages));
}
