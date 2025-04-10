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
