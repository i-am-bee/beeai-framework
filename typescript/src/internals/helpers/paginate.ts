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
