/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseCache } from "@/cache/base.js";

export class UnconstrainedCache<T> extends BaseCache<T> {
  protected readonly provider = new Map<string, T>();

  static {
    this.register();
  }

  async get(key: string) {
    return this.provider.get(key);
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
    this.provider.set(key, value);
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
