/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "@/internals/serializable.js";
import { Callback } from "@/emitter/types.js";
import { FrameworkError } from "@/errors.js";
import { Emitter } from "@/emitter/emitter.js";
import { shallowCopy } from "@/serializer/utils.js";
import { GetRunContext, RunContext } from "@/context.js";
import { pRetry } from "@/internals/helpers/retry.js";
import { FullModelName, loadModel, parseModel } from "@/backend/utils.js";
import { ProviderName } from "@/backend/constants.js";
import { EmbeddingModelError } from "@/backend/errors.js";

export interface EmbeddingModelInput {
  values: string[];
  abortSignal?: AbortSignal;
  maxRetries?: number;
}

export interface EmbeddingModelOutput {
  values: string[];
  embeddings: number[][];
  usage: { tokens?: number };
}

export interface EmbeddingModelEvents {
  success?: Callback<{ value: EmbeddingModelOutput }>;
  start?: Callback<{ input: EmbeddingModelInput }>;
  error?: Callback<{ input: EmbeddingModelInput; error: FrameworkError }>;
  finish?: Callback<null>;
}

export type EmbeddingModelEmitter<A = Record<never, never>> = Emitter<
  EmbeddingModelEvents & Omit<A, keyof EmbeddingModelEvents>
>;

export abstract class EmbeddingModel extends Serializable {
  public abstract readonly emitter: Emitter<EmbeddingModelEvents>;

  abstract get modelId(): string;
  abstract get providerId(): string;

  create(input: EmbeddingModelInput) {
    input = shallowCopy(input);

    return RunContext.enter(
      this,
      { params: [input] as const, signal: input?.abortSignal },
      async (run) => {
        try {
          await run.emitter.emit("start", { input });
          const result: EmbeddingModelOutput = await pRetry(() => this._create(input, run), {
            retries: input.maxRetries || 0,
            signal: run.signal,
          });
          await run.emitter.emit("success", { value: result });
          return result;
        } catch (error) {
          await run.emitter.emit("error", { input, error });
          if (error instanceof EmbeddingModelError) {
            throw error;
          } else {
            throw new EmbeddingModelError(`The Embedding Model has encountered an error.`, [error]);
          }
        } finally {
          await run.emitter.emit("finish", null);
        }
      },
    );
  }

  static async fromName(name: FullModelName | ProviderName) {
    const { providerId, modelId = "" } = parseModel(name);
    const Target = await loadModel<EmbeddingModel>(providerId, "embedding");
    return new Target(modelId);
  }

  protected abstract _create(
    input: EmbeddingModelInput,
    run: GetRunContext<typeof this>,
  ): Promise<EmbeddingModelOutput>;

  createSnapshot() {
    return { emitter: this.emitter };
  }

  destroy() {
    this.emitter.destroy();
  }
}
