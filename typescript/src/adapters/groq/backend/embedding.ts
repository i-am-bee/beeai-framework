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
