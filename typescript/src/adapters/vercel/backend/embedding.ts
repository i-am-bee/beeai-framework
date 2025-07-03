/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  EmbeddingModel,
  EmbeddingModelInput,
  EmbeddingModelOutput,
  EmbeddingModelEvents,
} from "@/backend/embedding.js";
import { embedMany, EmbeddingModel as Model } from "ai";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext } from "@/context.js";
import { toCamelCase } from "remeda";
import { FullModelName } from "@/backend/utils.js";

type InternalEmbeddingModel = Model<string>;

export class VercelEmbeddingModel<
  R extends InternalEmbeddingModel = InternalEmbeddingModel,
> extends EmbeddingModel {
  public readonly emitter: Emitter<EmbeddingModelEvents>;

  constructor(public readonly model: R) {
    super();
    this.emitter = Emitter.root.child({
      namespace: ["backend", this.providerId, "embedding"],
      creator: this,
    });
  }

  get modelId(): string {
    return this.model.modelId;
  }

  get providerId(): string {
    const provider = this.model.provider.split(".")[0].split("-")[0];
    return toCamelCase(provider);
  }

  protected async _create(
    input: EmbeddingModelInput,
    run: GetRunContext<this>,
  ): Promise<EmbeddingModelOutput> {
    return embedMany<string>({
      model: this.model,
      values: input.values,
      abortSignal: run.signal,
    });
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      providerId: this.providerId,
      modelId: this.model,
    };
  }

  async loadSnapshot({ providerId, modelId, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    const instance = await VercelEmbeddingModel.fromName(
      `${providerId}:${modelId}` as FullModelName,
    );
    if (!(instance instanceof VercelEmbeddingModel)) {
      throw new Error("Incorrect deserialization!");
    }
    instance.destroy();
    Object.assign(this, {
      ...snapshot,
      model: instance.model,
    });
  }
}
