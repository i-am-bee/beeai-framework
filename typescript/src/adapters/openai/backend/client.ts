/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createOpenAI, OpenAIProvider, OpenAIProviderSettings } from "@ai-sdk/openai";
import { getEnv } from "@/internals/env.js";
import { BackendClient } from "@/backend/client.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type OpenAIClientSettings = OpenAIProviderSettings;

export class OpenAIClient extends BackendClient<OpenAIClientSettings, OpenAIProvider> {
  protected create(): OpenAIProvider {
    return createOpenAI({
      ...this.settings,
      apiKey: this.settings?.apiKey || getEnv("OPENAI_API_KEY"),
      baseURL: this.settings?.baseURL || getEnv("OPENAI_API_ENDPOINT"),
      headers: {
        ...parseHeadersFromEnv("OPENAI_API_HEADERS"),
        ...this.settings?.headers,
      },
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
