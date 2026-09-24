/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleVertexClient, GoogleVertexClientSettings } from "./client.js";
import { VercelEmbeddingModel } from "@/adapters/vercel/backend/embedding.js";
import { getEnv } from "@/internals/env.js";
import { GoogleVertexProvider } from "@ai-sdk/google-vertex";

type GoogleVertexParameters = Parameters<GoogleVertexProvider["textEmbeddingModel"]>;
export type GoogleVertexChatModelId = NonNullable<GoogleVertexParameters[0]>;
export type GoogleVertexChatModelSettings = Record<string, any>;

export class GoogleVertexEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: GoogleVertexChatModelId = getEnv(
      "GOOGLE_VERTEX_EMBEDDING_MODEL",
      "text-embedding-004",
    ),
    _settings: GoogleVertexChatModelSettings = {},
    client?: GoogleVertexClient | GoogleVertexClientSettings,
  ) {
    const model = GoogleVertexClient.ensure(client).instance.textEmbeddingModel(modelId);
    super(model);
  }
}
