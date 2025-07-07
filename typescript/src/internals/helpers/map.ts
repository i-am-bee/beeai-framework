/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export function findUniqueKey(baseKey: string, map: Map<string, unknown>) {
  let key = baseKey;
  for (let i = 1; map.has(key); i++) {
    key = baseKey.concat(String(i));
  }
  return key;
}
