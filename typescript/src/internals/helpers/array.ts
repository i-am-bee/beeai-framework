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

interface ArrayChangeHandlers<T> {
  onAdd?: (value: T) => void;
  onRemove?: (value: T) => void;
}

export function watchArray<T>(array: T[], handlers: ArrayChangeHandlers<T>): T[] {
  /**
   * Does not handle 'length' modification
   */

  return new Proxy(array, {
    set(target, property, value, receiver) {
      const apply = () => Reflect.set(target, property, value, receiver);

      const index = Number(property);

      // Ignore non-index and length changes
      if (!Number.isInteger(index) || index < 0) {
        return apply();
      }

      const isAdd = !(property in target);
      const oldValue = target[index];
      const result = apply();

      if (isAdd) {
        handlers.onAdd?.(value);
      } else if (oldValue !== value) {
        // Treat replacement as removal + add
        handlers.onRemove?.(oldValue);
        handlers.onAdd?.(value);
      }

      return result;
    },

    deleteProperty(target, property) {
      const index = Number(property);

      if (!Number.isInteger(index) || index < 0) {
        return Reflect.deleteProperty(target, property);
      }

      const oldValue = target[index];
      const result = Reflect.deleteProperty(target, property);

      if (result && oldValue !== undefined) {
        handlers.onRemove?.(oldValue);
      }

      return result;
    },
  });
}
