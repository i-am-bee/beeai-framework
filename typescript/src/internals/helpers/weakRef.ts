/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export function isWeakKey(value: unknown): value is WeakKey {
  return value != null && typeof value === "object";
}

export class SafeWeakSet extends WeakSet {
  has(value: unknown) {
    if (isWeakKey(value)) {
      return super.has(value);
    }
    return false;
  }

  add(value: unknown) {
    if (isWeakKey(value)) {
      super.add(value);
    }
    return this;
  }

  delete(value: unknown) {
    if (isWeakKey(value)) {
      return super.delete(value);
    }
    return false;
  }
}

export class SafeWeakMap<T> extends WeakMap<WeakKey, T> {
  has(value: unknown) {
    if (isWeakKey(value)) {
      return super.has(value);
    }
    return false;
  }

  get(key: unknown) {
    if (isWeakKey(key)) {
      return super.get(key);
    }
  }

  set(key: unknown, value: T) {
    if (isWeakKey(key)) {
      super.set(key, value);
    }
    return this;
  }

  delete(value: unknown) {
    if (isWeakKey(value)) {
      return super.delete(value);
    }
    return false;
  }
}
