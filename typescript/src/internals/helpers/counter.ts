/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";
import { Serializable } from "@/internals/serializable.js";

export class RetryCounter extends Serializable {
  public remaining: number;
  protected readonly maxRetries: number;
  protected lastError?: Error;
  protected finalError?: Error;

  constructor(
    maxRetries = 0,
    protected ErrorClass: typeof FrameworkError,
  ) {
    super();
    this.maxRetries = maxRetries;
    this.remaining = maxRetries;
  }

  use(error?: Error) {
    if (this.finalError) {
      throw this.finalError;
    }

    this.lastError = error ?? this.lastError;
    this.remaining--;
    if (this.remaining < 0) {
      this.finalError = new this.ErrorClass(
        `Maximal amount of global retries (${this.maxRetries}) has been reached.`,
        this.lastError ? [this.lastError] : undefined,
        { isFatal: true, isRetryable: false },
      );
      throw this.finalError;
    }
  }

  createSnapshot() {
    return {
      remaining: this.remaining,
      maxRetries: this.maxRetries,
      lastError: this.lastError,
      finalError: this.finalError,
      ErrorClass: this.ErrorClass,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
