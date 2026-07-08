/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
