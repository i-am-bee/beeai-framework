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
