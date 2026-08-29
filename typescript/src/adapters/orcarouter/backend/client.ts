/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createOpenAI, OpenAIProvider, OpenAIProviderSettings } from "@ai-sdk/openai";
import { getEnv } from "@/internals/env.js";
import { BackendClient } from "@/backend/client.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

const ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1";

export type OrcaRouterClientSettings = OpenAIProviderSettings;

export class OrcaRouterClient extends BackendClient<OrcaRouterClientSettings, OpenAIProvider> {
  protected create(): OpenAIProvider {
    return createOpenAI({
      ...this.settings,
      apiKey: this.settings?.apiKey || getEnv("ORCAROUTER_API_KEY"),
      baseURL: this.settings?.baseURL || getEnv("ORCAROUTER_API_BASE", ORCAROUTER_API_BASE),
      headers: {
        ...parseHeadersFromEnv("ORCAROUTER_API_HEADERS"),
        ...this.settings?.headers,
      },
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
