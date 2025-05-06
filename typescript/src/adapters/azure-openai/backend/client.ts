/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
