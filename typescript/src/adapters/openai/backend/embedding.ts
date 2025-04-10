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
