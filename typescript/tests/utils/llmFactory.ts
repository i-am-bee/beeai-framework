/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import process from "node:process";
import { Agent, Dispatcher } from "undici";
import { OpenAIChatModel } from "@/adapters/openai/backend/chat.js";
import { OllamaChatModel } from "@/adapters/ollama/backend/chat.js";
import { WatsonxChatModel } from "@/adapters/watsonx/backend/chat.js";
import { ChatModel } from "@/backend/chat.js";
import { GroqChatModel } from "@/adapters/groq/backend/chat.js";
import { AzureOpenAIChatModel } from "@/adapters/azure-openai/backend/chat.js";

export function createChatLLM(): ChatModel {
  if (process.env.OPENAI_API_KEY) {
    return new OpenAIChatModel("gpt-4o");
  } else if (process.env.AZURE_OPENAI_API_KEY) {
    return new AzureOpenAIChatModel("gpt-4o");
  } else if (process.env.WATSONX_API_KEY && process.env.WATSONX_PROJECT_ID) {
    return new WatsonxChatModel("meta-llama/llama-3-3-70b-instruct");
  } else if (process.env.GROQ_API_KEY) {
    return new GroqChatModel(`llama-3.3-70b-versatile`);
  } else if (process.env.OLLAMA_BASE_URL) {
    // the undici definition of RequestInit does not extend the default
    // fetch RequestInit so we can't use its type directly. Define
    // and interface that adds the field we need to the default fetch
    // interface to that we can make TypeScript accept it.
    interface UndiciRequestInit extends RequestInit {
      dispatcher: Dispatcher;
    }

    return new OllamaChatModel(
      "llama3.1:8b",
      {},
      {
        baseURL: process.env.OLLAMA_BASE_URL,
        fetch: (input, init?) => {
          const someInit = init || {};
          const requestInit: UndiciRequestInit = {
            ...someInit,
            dispatcher: new Agent({ headersTimeout: 2700000 }),
          };
          return fetch(input, requestInit);
        },
      },
    );
  } else {
    throw new Error("No API key for any LLM provider has been provided. Cannot run test case.");
  }
}
