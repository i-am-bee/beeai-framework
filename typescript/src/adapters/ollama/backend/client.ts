/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { getEnv } from "@/internals/env.js";
import { createOllama, OllamaProvider, OllamaProviderSettings } from "ollama-ai-provider";
import { BackendClient } from "@/backend/client.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type OllamaClientSettings = OllamaProviderSettings & { apiKey?: string };

export class OllamaClient extends BackendClient<OllamaClientSettings, OllamaProvider> {
  protected create(): OllamaProvider {
    const { apiKey: _apiKey, baseURL, headers, ...settings } = this.settings ?? {};
    const apiKey = _apiKey || getEnv("OLLAMA_API_KEY");

    return createOllama({
      ...settings,
      baseURL: baseURL || getEnv("OLLAMA_BASE_URL"),
      fetch: vercelFetcher(this.settings?.fetch),
      headers: {
        ...parseHeadersFromEnv("OLLAMA_API_HEADERS"),
        ...headers,
        ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
      },
    });
  }
}
