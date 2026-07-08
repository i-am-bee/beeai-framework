/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "@/internals/serializable.js";

export abstract class BaseCache<T> extends Serializable {
  public enabled = true;

  abstract size(): Promise<number>;
  abstract set(key: string, value: T): Promise<void>;
  abstract get(key: string): Promise<T | undefined>;
  abstract has(key: string): Promise<boolean>;
  abstract delete(key: string): Promise<boolean>;
  abstract clear(): Promise<void>;
}
