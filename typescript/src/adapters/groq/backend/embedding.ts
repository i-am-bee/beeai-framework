/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { getEnv } from "@/internals/env.js";
import { GroqClient, GroqClientSettings } from "@/adapters/groq/backend/client.js";
import { ValueError } from "@/errors.js";
import { GroqProvider } from "@ai-sdk/groq";

type GroqParameters = Parameters<GroqProvider["textEmbeddingModel"]>;
export type GroqEmbeddingModelId = NonNullable<GroqParameters[0]>;
export type GroqEmbeddingModelSettings = Record<string, any>;

export class GroqEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: GroqEmbeddingModelId = getEnv("GROQ_EMBEDDING_MODEL", ""),
    _settings: GroqEmbeddingModelSettings = {},
    client?: GroqClientSettings | GroqClient,
  ) {
    if (!modelId) {
      throw new ValueError("Missing modelId!");
    }
    const model = GroqClient.ensure(client).instance.textEmbeddingModel(modelId);
    super(model);
  }
}
