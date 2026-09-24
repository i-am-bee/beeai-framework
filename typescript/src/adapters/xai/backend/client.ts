/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createXai, type XaiProvider, type XaiProviderSettings } from "@ai-sdk/xai";
import { BackendClient } from "@/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type XaiClientSettings = XaiProviderSettings;

export class XaiClient extends BackendClient<XaiClientSettings, XaiProvider> {
  protected create(): XaiProvider {
    return createXai({
      ...this.settings,
      baseURL: this.settings.baseURL || getEnv("XAI_API_BASE_URL"),
      apiKey: this.settings.apiKey || getEnv("XAI_API_KEY"),
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
