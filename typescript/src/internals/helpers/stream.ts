/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export async function* transformAsyncIterable<A, B, A2>(
  generator: AsyncGenerator<A, B> | AsyncIterableIterator<A>,
  transformer: (old: A) => A2,
): AsyncGenerator<A2, B> {
  let next: IteratorResult<A, B>;
  while (!(next = await generator.next()).done) {
    yield transformer(next.value);
  }
  return next.value;
}

export function isAsyncIterable<T>(value: any): value is AsyncIterable<T> {
  return Boolean(value && Symbol.asyncIterator in value);
}
