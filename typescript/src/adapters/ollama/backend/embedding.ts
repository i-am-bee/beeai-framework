/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OllamaProvider } from "ollama-ai-provider";
import { OllamaClient, OllamaClientSettings } from "@/adapters/ollama/backend/client.js";
import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { getEnv } from "@/internals/env.js";

type OllamaParameters = Parameters<OllamaProvider["textEmbeddingModel"]>;
export type OllamaEmbeddingModelId = NonNullable<OllamaParameters[0]>;
export type OllamaEmbeddingModelSettings = NonNullable<OllamaParameters[1]>;

export class OllamaEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: OllamaEmbeddingModelId = getEnv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    settings: OllamaEmbeddingModelSettings = {},
    client?: OllamaClient | OllamaClientSettings,
  ) {
    const model = OllamaClient.ensure(client).instance.embedding(modelId, settings);
    super(model);
  }
}
