/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as R from "remeda";
import { Task, TaskState } from "promise-based-task";
import { FrameworkError } from "@/errors.js";
import { EnumValue } from "@/internals/types.js";
import { asyncProperties } from "@/internals/helpers/promise.js";
import { createRandomHash } from "@/internals/helpers/hash.js";
import { pRetry } from "@/internals/helpers/retry.js";

export interface RetryableConfig {
  maxRetries: number;
  factor?: number;
  signal?: AbortSignal;
}

export const RunStrategy = {
  /**
   * Once a single Retryable throws, other retry ables get cancelled immediately.
   */
  THROW_IMMEDIATELY: "THROW_IMMEDIATELY",

  /**
   * Once a single Retryable throws, wait for other to completes, but prevent further retries.
   */
  SETTLE_ROUND: "SETTLE_ROUND",

  /**
   * Once a single Retryable throws, other Retryables remains to continue. Error is thrown by the end.
   */
  SETTLE_ALL: "SETTLE_ALL",
} as const;

export interface RetryableRunConfig {
  groupSignal: AbortSignal;
}

export interface RetryableContext {
  executionId: string;
  attempt: number;
  signal?: AbortSignal;
}

export type RetryableHandler<T> = (ctx: RetryableContext) => Promise<T>;
export type ResetHandler = () => void;
export type ErrorHandler = (error: Error, ctx: RetryableContext) => void | Promise<void>;
export type RetryHandler = (ctx: RetryableContext, lastError: Error) => void | Promise<void>;

export class Retryable<T> {
  readonly #id: string;
  #value: Task<T, Error> | null;
  #config: RetryableConfig;
  #handlers;

  constructor(ctx: {
    executor: RetryableHandler<T>;
    onReset?: ResetHandler;
    onError?: ErrorHandler;
    onRetry?: RetryHandler;
    config?: Partial<RetryableConfig>;
  }) {
    this.#value = null;
    this.#id = createRandomHash();
    this.#handlers = {
      executor: ctx.executor,
      onReset: ctx.onReset,
      onError: ctx.onError,
      onRetry: ctx.onRetry,
    } as const;
    this.#config = {
      ...ctx.config,
      maxRetries: Math.max(ctx.config?.maxRetries || 0, 0),
    };
  }

  static async runGroup<T>(
    strategy: EnumValue<typeof RunStrategy>,
    inputs: Retryable<T>[],
  ): Promise<T[]> {
    if (strategy === RunStrategy.THROW_IMMEDIATELY) {
      return await Promise.all(inputs.map((input) => input.get()));
    }

    const controller = new AbortController();
    const results = await Promise.allSettled(
      inputs.map((input) =>
        input
          .get(strategy === RunStrategy.SETTLE_ALL ? { groupSignal: controller.signal } : undefined)
          .catch((err) => {
            controller.abort(err);
            throw err;
          }),
      ),
    );
    controller.signal.throwIfAborted();
    return results.map((result) => (result as PromiseFulfilledResult<T>).value!);
  }

  static async *runSequence<T>(inputs: readonly Retryable<T>[]): AsyncGenerator<T> {
    for (const input of inputs) {
      yield await input.get();
    }
  }

  static async collect<T>(inputs: T & Record<string, Retryable<any>>) {
    // Solve everything
    await Promise.all(R.values(inputs).map((input) => input.get()));

    // Obtain latest values
    return await asyncProperties(
      R.mapValues(inputs, (value) => (value as Retryable<any>).get()) as {
        [K in keyof T]: Promise<T[K] extends Retryable<infer Q> ? Q : never>;
      },
    );
  }

  #getContext(attempt: number): RetryableContext {
    const ctx: RetryableContext = {
      attempt,
      executionId: this.#id,
      signal: this.#config.signal,
    };
    Object.defineProperty(ctx, "signal", {
      enumerable: false,
    });
    return ctx;
  }

  get isResolved(): boolean {
    return this.#value?.state === TaskState.RESOLVED;
  }

  get isRejected(): boolean {
    return this.#value?.state === TaskState.REJECTED;
  }

  protected _run(config?: RetryableRunConfig) {
    const task = new Task<T, Error>();

    const assertAborted = () => {
      this.#config.signal?.throwIfAborted?.();
      config?.groupSignal?.throwIfAborted?.();
    };

    let lastError: Error | null = null;
    pRetry(
      async (attempt) => {
        assertAborted();

        const ctx = this.#getContext(attempt);
        if (attempt > 1) {
          await this.#handlers.onRetry?.(ctx, lastError!);
        }
        return await this.#handlers.executor(ctx);
      },
      {
        retries: this.#config.maxRetries,
        factor: this.#config.factor,
        signal: this.#config.signal,
        shouldRetry: (e) => {
          if (!FrameworkError.isRetryable(e)) {
            return false;
          }
          return !config?.groupSignal?.aborted && !this.#config.signal?.aborted;
        },
        onFailedAttempt: async (e, meta) => {
          lastError = e;
          await this.#handlers.onError?.(e, this.#getContext(meta.attempt));
          if (!FrameworkError.isRetryable(e)) {
            throw e;
          }
          assertAborted();
        },
      },
    )
      .then((x) => task.resolve(x))
      .catch((x) => task.reject(x));

    return task;
  }

  async get(config?: RetryableRunConfig): Promise<T> {
    if (this.isResolved) {
      return this.#value!.resolvedValue()!;
    }
    if (this.isRejected) {
      throw this.#value?.rejectedValue();
    }

    if (this.#value?.state === TaskState.PENDING && !config) {
      return this.#value;
    }

    this.#value?.catch?.(() => {});
    this.#value = this._run(config);
    return this.#value;
  }

  reset() {
    this.#value?.catch?.(() => {});
    this.#value = null;
    this.#handlers.onReset?.();
  }
}
