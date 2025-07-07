/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { AmazonBedrockProvider } from "@ai-sdk/amazon-bedrock";
import { getEnv } from "@/internals/env.js";
import {
  AmazonBedrockClient,
  AmazonBedrockClientSettings,
} from "@/adapters/amazon-bedrock/backend/client.js";

type Params = Parameters<AmazonBedrockProvider["embedding"]>;
export type BedrockEmbeddingModelId = NonNullable<Params[0]>;
export type BedrockEmbeddingSettings = NonNullable<Params[1]>;

export class BedrockEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: BedrockEmbeddingModelId = getEnv("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1"),
    settings: BedrockEmbeddingSettings = {},
    client?: AmazonBedrockClient | AmazonBedrockClientSettings,
  ) {
    const model = AmazonBedrockClient.ensure(client).instance.embedding(modelId, settings);
    super(model);
  }
}
