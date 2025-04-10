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

import { SlidingTaskMap, Task } from "promise-based-task";
import { BaseCache } from "@/cache/base.js";
import { Serializable } from "@/internals/serializable.js";

export interface SlidingCacheInput {
  size: number;
  ttl?: number;
}

class SlidingCacheEntry<T> extends Serializable {
  constructor(protected readonly value: T) {
    super();
  }

  static {
    this.register();
  }

  destructor() {
    if (this.value instanceof Task) {
      this.value.destructor();
    }
  }

  unwrap(): T {
    return this.value;
  }

  createSnapshot() {
    return { value: this.value };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

export class SlidingCache<T> extends BaseCache<T> {
  protected readonly provider;

  constructor(input: SlidingCacheInput) {
    super();
    this.provider = new SlidingTaskMap<string, SlidingCacheEntry<T>>(input.size, input.ttl);
  }

  static {
    this.register();
  }

  async get(key: string) {
    const value = this.provider.get(key);
    return value?.unwrap?.();
  }

  async has(key: string) {
    return this.provider.has(key);
  }

  async clear() {
    this.provider.clear();
  }

  async delete(key: string) {
    return this.provider.delete(key);
  }

  async set(key: string, value: T) {
    this.provider.set(key, new SlidingCacheEntry(value));
  }

  async size() {
    return this.provider.size;
  }

  createSnapshot() {
    return {
      enabled: this.enabled,
      provider: this.provider,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
