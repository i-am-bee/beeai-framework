/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { OllamaProvider } from "ollama-ai-provider";
import { OllamaClient, OllamaClientSettings } from "@/adapters/ollama/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelToolChoiceSupport } from "@/backend/chat.js";

type OllamaParameters = Parameters<OllamaProvider["languageModel"]>;
export type OllamaChatModelId = NonNullable<OllamaParameters[0]>;
export type OllamaChatModelSettings = NonNullable<OllamaParameters[1]>;

export class OllamaChatModel extends VercelChatModel {
  readonly supportsToolStreaming = false;
  public readonly toolChoiceSupport: ChatModelToolChoiceSupport[] = ["none", "auto"];

  constructor(
    modelId: OllamaChatModelId = getEnv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
    settings: OllamaChatModelSettings = {},
    client?: OllamaClient | OllamaClientSettings,
  ) {
    const model = OllamaClient.ensure(client).instance.chat(modelId, {
      ...settings,
      structuredOutputs: true, // otherwise breaks generated structure
    });
    super(model);
  }

  static {
    this.register();
  }
}
