/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIClient } from "@/adapters/openai/backend/client.js";
import { OpenAIProvider, OpenAIProviderSettings } from "@ai-sdk/openai";
import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { getEnv } from "@/internals/env.js";

type OpenAIParameters = Parameters<OpenAIProvider["embedding"]>;
export type OpenAIEmbeddingModelId = NonNullable<OpenAIParameters[0]>;
export type OpenAIEmbeddingModelSettings = NonNullable<OpenAIParameters[1]>;

export class OpenAIEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: OpenAIEmbeddingModelId = getEnv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    settings: OpenAIEmbeddingModelSettings = {},
    client?: OpenAIProviderSettings | OpenAIClient,
  ) {
    const model = OpenAIClient.ensure(client).instance.embedding(modelId, settings);
    super(model);
  }
}
