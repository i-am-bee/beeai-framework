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

export function removeFromArray<T>(arr: T[], target: T): boolean {
  const index = arr.findIndex((value) => value === target);
  if (index === -1) {
    return false;
  }

  arr.splice(index, 1);
  return true;
}

export function castArray<T>(arr: T) {
  const result = Array.isArray(arr) ? arr : [arr];
  return result as T extends unknown[] ? T : [T];
}

type HasMinLength<T, N extends number, T2 extends any[] = []> = T2["length"] extends N
  ? [...T2, ...T[]]
  : HasMinLength<T, N, [any, ...T2]>;

export function hasMinLength<T, N extends number>(arr: T[], n: N): arr is HasMinLength<T, N> {
  return arr.length >= n;
}
