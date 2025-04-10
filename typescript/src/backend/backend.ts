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

import { Serializable } from "@/internals/serializable.js";
import { ChatModel } from "@/backend/chat.js";
import { EmbeddingModel } from "@/backend/embedding.js";
import { asyncProperties } from "@/internals/helpers/promise.js";
import { FullModelName } from "@/backend/utils.js";
import { ProviderName } from "./constants.js";
import { OptionalExcept } from "@/internals/types.js";

interface BackendModels {
  chat: ChatModel;
  embedding: EmbeddingModel;
}

export class Backend extends Serializable implements BackendModels {
  chat!: ChatModel;
  embedding!: EmbeddingModel;

  constructor(models: BackendModels) {
    super();
    Object.assign(this, models);
  }

  static async fromName(
    input: OptionalExcept<Record<keyof BackendModels, FullModelName | ProviderName>, "chat">,
  ): Promise<Backend> {
    return new Backend(
      await asyncProperties({
        chat: ChatModel.fromName(input.chat),
        embedding: EmbeddingModel.fromName(input.embedding || "dummy"),
      }),
    );
  }

  static async fromProvider(provider: ProviderName): Promise<Backend> {
    return await this.fromName({
      chat: provider,
      embedding: provider,
    });
  }

  createSnapshot() {
    return { chat: this.chat, embedding: this.embedding };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
