/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createAnthropic, AnthropicProvider, AnthropicProviderSettings } from "@ai-sdk/anthropic";
import { BackendClient } from "@/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type AnthropicClientSettings = AnthropicProviderSettings;

export class AnthropicClient extends BackendClient<AnthropicClientSettings, AnthropicProvider> {
  protected create(): AnthropicProvider {
    const extraHeaders = parseHeadersFromEnv("ANTHROPIC_API_HEADERS");

    return createAnthropic({
      ...this.settings,
      baseURL: this.settings?.baseURL || getEnv("ANTHROPIC_API_BASE_URL"),
      apiKey: this.settings.apiKey || getEnv("ANTHROPIC_API_KEY"),
      headers: {
        ...extraHeaders,
        ...this.settings?.headers,
      },
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
