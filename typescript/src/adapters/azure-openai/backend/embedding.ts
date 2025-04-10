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

import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import {
  AzureOpenAIClient,
  AzureOpenAIClientSettings,
} from "@/adapters/azure-openai/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { AzureOpenAIProvider as VercelAzureOpenAIProviderSettings } from "@ai-sdk/azure";

type AzureOpenAIParameters = Parameters<VercelAzureOpenAIProviderSettings["textEmbeddingModel"]>;
export type AzureOpenAIEmbeddingModelId = NonNullable<AzureOpenAIParameters[0]>;
export type AzureOpenAIEmbeddingModelSettings = Record<string, any>;

export class AzureOpenAIEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: AzureOpenAIEmbeddingModelId = getEnv(
      "AZURE_OPENAI_EMBEDDING_MODEL",
      "text-embedding-3-small",
    ),
    settings: AzureOpenAIEmbeddingModelSettings = {},
    client?: AzureOpenAIClient | AzureOpenAIClientSettings,
  ) {
    const model = AzureOpenAIClient.ensure(client).instance.textEmbeddingModel(modelId, settings);
    super(model);
  }
}
