/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createGroq, GroqProvider, GroqProviderSettings } from "@ai-sdk/groq";
import { BackendClient } from "@/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type GroqClientSettings = GroqProviderSettings;

export class GroqClient extends BackendClient<GroqClientSettings, GroqProvider> {
  protected create(): GroqProvider {
    return createGroq({
      ...this.settings,
      baseURL: this.settings.baseURL || getEnv("GROQ_API_BASE_URL"),
      apiKey: this.settings.apiKey || getEnv("GROQ_API_KEY"),
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
