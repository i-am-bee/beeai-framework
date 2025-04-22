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
import { shallowCopy } from "@/serializer/utils.js";

export abstract class BackendClient<P, T> extends Serializable {
  public readonly instance: T;
  protected readonly settings: P;

  constructor(settings: P) {
    super();
    this.settings = settings ?? ({} as P);
    this.instance = this.create();
  }

  protected abstract create(): T;

  createSnapshot() {
    return {
      settings: shallowCopy(this.settings),
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
    Object.assign(this, { instance: this.create() });
  }

  static ensure<P2, T2, R extends BackendClient<P2, T2>>(
    this: new (settings: P2) => R,
    settings?: P2 | R,
  ): R {
    if (settings && settings instanceof this) {
      return settings;
    }
    return new this(settings as P2);
  }
}
