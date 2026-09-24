/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as R from "remeda";

export type Primitive = string | number | symbol | bigint | boolean | null | undefined;
export function isPrimitive(value: unknown): value is Primitive {
  return (
    R.isString(value) ||
    R.isNumber(value) ||
    R.isBoolean(value) ||
    R.isNullish(value) ||
    R.isSymbol(value)
  );
}
