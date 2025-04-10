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
