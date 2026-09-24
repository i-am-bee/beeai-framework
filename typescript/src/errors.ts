/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import util from "util";
import * as R from "remeda";
import type { ObjectLike } from "@/internals/types.js";

export interface FrameworkErrorOptions {
  isFatal?: boolean;
  isRetryable?: boolean;
  context?: ObjectLike;
}

export class FrameworkError extends AggregateError {
  isFatal = false;
  isRetryable = true;
  context: Record<string, any>;

  constructor(
    message = "The BeeAI Framework has encountered an error.",
    errors: Error[] = [],
    options: FrameworkErrorOptions = {},
  ) {
    super(errors || [], message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, AggregateError);

    if (R.isBoolean(options?.isFatal)) {
      this.isFatal = options?.isFatal;
    }
    if (R.isBoolean(options?.isRetryable)) {
      this.isRetryable = options.isRetryable;
    }
    this.context = options?.context ?? {};
  }

  hasFatalError() {
    if (this.isFatal) {
      return true;
    }
    for (const err of this.traverseErrors()) {
      if (err instanceof FrameworkError && err.isFatal) {
        return true;
      }
    }
    return false;
  }

  *traverseErrors(): Generator<Error> {
    for (const error of this.errors) {
      yield error;
      if (error instanceof FrameworkError) {
        yield* error.traverseErrors();
      }
    }
  }

  getCause(): Error {
    const errors = Array.from(this.traverseErrors());
    return errors.at(-1) ?? this;
  }

  explain(): string {
    const output = [];
    for (const [index, error] of [this, ...this.traverseErrors()].entries()) {
      const offset = `  `.repeat(2 * index);
      output.push(`${offset}${error.toString()}`);
      if (error.cause) {
        output.push(`${offset}Cause: ${error.cause.toString()}`);
      }
    }
    return output.join("\n");
  }

  dump() {
    return util.inspect(this, {
      compact: false,
      depth: Infinity,
    });
  }

  static ensure(error: Error): FrameworkError {
    return error instanceof FrameworkError ? error : new FrameworkError(undefined, [error]);
  }

  static isInstanceOf(error: unknown): error is FrameworkError {
    return error instanceof FrameworkError;
  }

  static isAbortError(error: unknown) {
    return error instanceof AbortError || (error instanceof Error && error?.name === "AbortError");
  }

  static isRetryable(error: unknown) {
    if (error instanceof FrameworkError) {
      return error.isRetryable;
    }
    return !FrameworkError.isAbortError(error);
  }
}

export class NotImplementedError extends FrameworkError {
  constructor(message = "Not implemented!", errors?: Error[]) {
    super(message, errors, {
      isFatal: true,
      isRetryable: false,
    });
  }
}

export class ValueError extends FrameworkError {
  constructor(
    message = "The provided value is not supported!",
    errors: Error[] = [],
    options?: FrameworkErrorOptions,
  ) {
    super(message, errors, options);
  }
}

export class AbortError extends FrameworkError {
  constructor(
    message = "The operation has been aborted!",
    errors: Error[] = [],
    options?: FrameworkErrorOptions,
  ) {
    super(message, errors, options);
  }
}
