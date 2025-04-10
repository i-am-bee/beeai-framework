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
