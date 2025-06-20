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
import { XaiClient, type XaiClientSettings } from "@/adapters/xai/backend/client.js";
import { ValueError } from "@/errors.js";
import type { XaiProvider } from "@ai-sdk/xai";

type XaiParameters = Parameters<XaiProvider["textEmbeddingModel"]>;
export type XAIEmbeddingModelId = NonNullable<XaiParameters[0]>;
export type XAIEmbeddingModelSettings = Record<string, unknown>;

export class XAIEmbeddingModel extends VercelEmbeddingModel {
  constructor(
    modelId: XAIEmbeddingModelId = getEnv("XAI_EMBEDDING_MODEL", ""),
    _settings: XAIEmbeddingModelSettings = {},
    client?: XaiClientSettings | XaiClient,
  ) {
    if (!modelId) {
      throw new ValueError("Missing modelId!");
    }
    const model = XaiClient.ensure(client).instance.textEmbeddingModel(modelId);
    super(model);
  }
}
