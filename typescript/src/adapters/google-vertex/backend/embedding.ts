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
