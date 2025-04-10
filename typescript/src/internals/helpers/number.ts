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

import * as R from "remeda";

export function safeSum(...numbers: (number | undefined | null)[]): number {
  return R.sum(numbers?.filter(R.isNonNullish) || [0]);
}

export function takeBigger(...numbers: (number | undefined | null)[]): number {
  return Math.max(...(numbers?.filter(R.isNonNullish) || [0])) || 0;
}

export function ensureRange(value: number, options: { min?: number; max?: number }): number {
  return Math.max(options.min ?? -Infinity, Math.min(value, options.max ?? Infinity));
}
