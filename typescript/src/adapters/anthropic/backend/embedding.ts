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
