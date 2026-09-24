/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createOpenAI, OpenAIProvider, OpenAIProviderSettings } from "@ai-sdk/openai";
import { getEnv } from "@/internals/env.js";
import { BackendClient } from "@/backend/client.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

const MINIMAX_API_BASE = "https://api.minimax.io/v1";

export type MiniMaxClientSettings = OpenAIProviderSettings;

export class MiniMaxClient extends BackendClient<MiniMaxClientSettings, OpenAIProvider> {
  protected create(): OpenAIProvider {
    return createOpenAI({
      ...this.settings,
      apiKey: this.settings?.apiKey || getEnv("MINIMAX_API_KEY"),
      baseURL: this.settings?.baseURL || getEnv("MINIMAX_API_BASE", MINIMAX_API_BASE),
      headers: {
        ...parseHeadersFromEnv("MINIMAX_API_HEADERS"),
        ...this.settings?.headers,
      },
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
