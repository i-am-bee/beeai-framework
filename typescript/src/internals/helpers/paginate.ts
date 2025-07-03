/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export interface PaginateInput<T, C> {
  size: number;
  handler: (data: { cursor?: C; limit: number }) => Promise<{ data: T[]; nextCursor?: C }>;
}

export async function paginate<T, C = number>(input: PaginateInput<T, C>): Promise<T[]> {
  const acc: T[] = [];
  let cursor: C | undefined = undefined;
  while (acc.length < input.size) {
    const { data, nextCursor } = await input.handler({
      cursor,
      limit: input.size - acc.length,
    });
    acc.push(...data);

    if (nextCursor === undefined || data.length === 0) {
      break;
    }
    cursor = nextCursor;
  }

  if (acc.length > input.size) {
    acc.length = input.size;
  }

  return acc;
}
