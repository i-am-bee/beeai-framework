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
