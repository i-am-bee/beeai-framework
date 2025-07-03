/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { GroqClient, GroqClientSettings } from "@/adapters/groq/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { GroqProvider } from "@ai-sdk/groq";

type GroqParameters = Parameters<GroqProvider["languageModel"]>;
export type GroqChatModelId = NonNullable<GroqParameters[0]>;
export type GroqChatModelSettings = NonNullable<GroqParameters[1]>;

export class GroqChatModel extends VercelChatModel {
  constructor(
    modelId: GroqChatModelId = getEnv("GROQ_CHAT_MODEL", "gemma2-9b-it"),
    settings: GroqChatModelSettings = {},
    client?: GroqClientSettings | GroqClient,
  ) {
    const model = GroqClient.ensure(client).instance.languageModel(modelId, settings);
    super(model);
  }

  static {
    this.register();
  }
}
