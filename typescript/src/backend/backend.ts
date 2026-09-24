/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
