/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
