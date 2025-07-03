/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
