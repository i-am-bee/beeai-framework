/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  EmbeddingModel,
  EmbeddingModelEvents,
  EmbeddingModelInput,
  EmbeddingModelOutput,
} from "@/backend/embedding.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { Embeddings as LCEmbeddingModel } from "@langchain/core/embeddings";
import { signalRace } from "@/internals/helpers/promise.js";

export class LangChainEmbeddingModel extends EmbeddingModel {
  public readonly emitter: Emitter<EmbeddingModelEvents>;

  constructor(protected readonly lcEmbedding: LCEmbeddingModel) {
    super();
    this.emitter = Emitter.root.child({
      namespace: ["langchain", "backend", "embedding"],
      creator: this,
    });
  }

  get modelId(): string {
    return "langchain"; // TODO
  }

  get providerId(): string {
    return "langchain";
  }

  protected async _create(
    input: EmbeddingModelInput,
    run: RunContext<this>,
  ): Promise<EmbeddingModelOutput> {
    const embeddings = await signalRace(
      () => this.lcEmbedding.embedDocuments(input.values),
      run.signal,
    );

    return {
      values: input.values.slice(),
      embeddings,
      usage: { tokens: undefined },
    };
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      lcEmbedding: this.lcEmbedding,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
