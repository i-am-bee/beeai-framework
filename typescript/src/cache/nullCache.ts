/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseCache } from "@/cache/base.js";

export class NullCache<T> extends BaseCache<T> {
  enabled = false;

  async set(_key: string, _value: T) {}

  async get(_key: string) {
    return undefined;
  }

  async has(_key: string) {
    return false;
  }

  async delete(_key: string) {
    return true;
  }

  async clear() {}

  async size() {
    return 0;
  }

  createSnapshot() {
    return {
      enabled: this.enabled,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
