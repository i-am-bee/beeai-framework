/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
