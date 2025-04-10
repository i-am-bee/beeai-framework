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
