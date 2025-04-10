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

import { isFunction, isPromise } from "remeda";
import { getProp } from "@/internals/helpers/object.js";

export function isPromiseLike<T>(value: unknown): value is PromiseLike<T> {
  return isPromise(value) || isFunction(getProp(value, ["then"]));
}

export async function arrayFromAsync<A, B>(
  generator: AsyncGenerator<A, B> | AsyncIterableIterator<A> | AsyncIterable<A>,
  limit?: number,
) {
  limit = limit ?? Infinity;
  const results: A[] = [];
  for await (const chunk of generator) {
    if (results.length >= limit) {
      break;
    }
    results.push(chunk);
  }
  return results;
}

export type EmitterToGeneratorFn<T, R> = (data: { emit: (data: T) => void }) => Promise<R>;
export async function* emitterToGenerator<T, R>(fn: EmitterToGeneratorFn<T, R>) {
  interface Data {
    data?: T;
    error?: Error;
    done?: boolean;
  }

  const queue: Data[] = [];
  void fn({
    emit: (data: T) => queue.push({ data }),
  })
    .then(() => queue.push({ done: true }))
    .catch((error) => queue.push({ error, done: true }));

  while (true) {
    while (queue.length === 0) {
      await new Promise((resolve) => setImmediate(resolve));
    }

    const { data, done, error } = queue.shift()!;
    if (error) {
      throw error;
    }
    if (done) {
      break;
    }
    yield data!;
  }
}

export async function asyncProperties<T extends NonNullable<unknown>>(
  obj: T,
): Promise<{ [K in keyof T]: Awaited<T[K]> }> {
  return Object.fromEntries(
    await Promise.all(Object.entries(obj).map(async ([key, value]) => [key, await value])),
  );
}

interface SafeExecuteOptions<T> {
  handler: () => T;
  onSuccess?: (result: T) => void;
  onError?: (err: Error) => void;
}

export function safeExecute<T>({ handler, onError, onSuccess }: SafeExecuteOptions<T>): void {
  try {
    const result = handler();
    if (isPromiseLike<T>(result)) {
      result.then(
        (result) => onSuccess?.(result),
        (err) => onError?.(err),
      );
    } else {
      onSuccess?.(result);
    }
  } catch (e) {
    onError?.(e);
  }
}

export function asyncExecute<R>(handler: () => R): Promise<R> {
  return new Promise((onSuccess, onError) => {
    safeExecute({
      handler,
      onSuccess,
      onError,
    });
  });
}

export class LazyPromise<R> implements Promise<R> {
  constructor(protected readonly handler: () => Promise<R>) {}

  readonly [Symbol.toStringTag] = "Promise";

  protected async before(): Promise<void> {}

  then<TResult1 = R, TResult2 = never>(
    onfulfilled?: ((value: R) => PromiseLike<TResult1> | TResult1) | undefined | null,
    onrejected?: ((reason: any) => PromiseLike<TResult2> | TResult2) | undefined | null,
  ): Promise<TResult1 | TResult2> {
    return this.before().then(this.handler).then(onfulfilled).catch(onrejected);
  }

  catch<TResult = never>(
    onrejected?: ((reason: any) => PromiseLike<TResult> | TResult) | undefined | null,
  ): Promise<R | TResult> {
    return this.before().then(this.handler).then(undefined).catch(onrejected);
  }

  finally(onfinally?: (() => void) | undefined | null): Promise<R> {
    return this.before().then(this.handler).finally(onfinally);
  }
}

export async function signalRace<R>(
  fn: () => Promise<R>,
  signal?: AbortSignal,
  onAbort?: () => void,
): Promise<R> {
  return new Promise<R>((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }

    const signalFn = () => {
      onAbort?.();
      reject(signal?.reason);
    };
    signal?.addEventListener?.("abort", signalFn);
    fn()
      .then(resolve)
      .catch(reject)
      .finally(() => signal?.removeEventListener?.("abort", signalFn));
  });
}

export async function executeSequentially(tasks: (() => Promise<any>)[]): Promise<void> {
  for (const task of tasks) {
    await task();
  }
}

export async function* toAsyncGenerator<T>(promise: T): AsyncGenerator<Awaited<T>> {
  yield await promise;
}
