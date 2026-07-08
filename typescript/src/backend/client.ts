/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
