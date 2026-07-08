/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { getEnv } from "@/internals/env.js";
import { AnthropicClient, AnthropicClientSettings } from "@/adapters/anthropic/backend/client.js";
import { AnthropicProvider } from "@ai-sdk/anthropic";

type AnthropicParameters = Parameters<AnthropicProvider["textEmbeddingModel"]>;
export type AnthropicEmbeddingModelId = NonNullable<AnthropicParameters[0]>;
export type AnthropicEmbeddingModelSettings = Record<string, any>;

export class AnthropicEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: AnthropicEmbeddingModelId = getEnv("ANTHROPIC_EMBEDDING_MODEL", "voyage-3-large"),
    _settings: AnthropicEmbeddingModelSettings = {},
    client?: AnthropicClientSettings | AnthropicClient,
  ) {
    const model = AnthropicClient.ensure(client).instance.textEmbeddingModel(modelId);
    super(model);
  }
}
