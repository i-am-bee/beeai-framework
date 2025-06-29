/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import "dotenv/config";
import { UserMessage } from "beeai-framework/backend/message";
import { WatsonxChatModel } from "beeai-framework/adapters/watsonx/backend/chat";

const chatLLM = new WatsonxChatModel("meta-llama/llama-3-1-70b-instruct");

// Log every request
chatLLM.emitter.match("*", async (data, event) => {
  console.info(
    `Time: ${event.createdAt.toISOString().substring(11, 19)}`,
    `Event: ${event.name}`,
    `Data: ${JSON.stringify(data).substring(0, 128).concat("...")}`,
  );
});

const response = await chatLLM.create({
  messages: [new UserMessage("Hello world!")],
});
console.info(response.messages[0]);
