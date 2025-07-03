/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ClassConstructor } from "@/internals/types.js";
import * as R from "remeda";

export function* traversePrototypeChain(value: unknown, excluded?: Set<any>) {
  let node = value;
  while (node !== null && node !== undefined) {
    node = Object.getPrototypeOf(node);
    if (!excluded?.has?.(node)) {
      yield node;
    }
  }
}

export function isDirectInstanceOf<A>(
  object: unknown,
  constructor: ClassConstructor<A>,
): object is A {
  return R.isTruthy(object) && Object.getPrototypeOf(object) === constructor.prototype;
}

export function findDescriptor<T extends NonNullable<unknown>>(
  target: T,
  property: keyof T,
): PropertyDescriptor | null {
  for (const node of traversePrototypeChain(target)) {
    const descriptor = Object.getOwnPropertyDescriptor(node, property);
    if (descriptor) {
      return descriptor;
    }
  }
  return null;
}
