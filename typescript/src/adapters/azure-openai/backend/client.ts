/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AzureOpenAIProvider, AzureOpenAIProviderSettings, createAzure } from "@ai-sdk/azure";
import { getEnv } from "@/internals/env.js";
import { BackendClient } from "@/backend/client.js";
import { vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type AzureOpenAIClientSettings = AzureOpenAIProviderSettings;

export class AzureOpenAIClient extends BackendClient<
  AzureOpenAIClientSettings,
  AzureOpenAIProvider
> {
  protected create(): AzureOpenAIProvider {
    return createAzure({
      ...this.settings,
      apiKey: this.settings.apiKey || getEnv("AZURE_OPENAI_API_KEY"),
      baseURL: this.settings.baseURL || getEnv("AZURE_OPENAI_API_ENDPOINT"),
      resourceName: this.settings.resourceName || getEnv("AZURE_OPENAI_API_RESOURCE"),
      apiVersion: this.settings.apiVersion || getEnv("AZURE_OPENAI_API_VERSION"),
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
