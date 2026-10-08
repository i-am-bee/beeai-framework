/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import dotenv from "dotenv";
import util from "util";
import { getFn, setFn } from "@vitest/runner";
import { FrameworkError } from "@/errors.js";
import { hasProps } from "@/internals/helpers/object.js";
dotenv.config();
dotenv.config({
  path: ".env.test",
  override: true,
});
dotenv.config({
  path: ".env.test.local",
  override: true,
});

function isFrameworkErrorLike(error: unknown): error is Record<keyof FrameworkError, any> {
  const keys = ["errors", "context", "isRetryable", "isFatal"] as (keyof FrameworkError)[];
  return hasProps(keys)(error as Record<keyof FrameworkError, any>);
}

function toReadableError(error: Record<keyof FrameworkError, any>): Error {
  const message = util
    .inspect(
      {
        message: error.message,
        context: error.context,
        cause: error.cause,
        isFatal: error.isFatal,
        isRetryable: error.isRetryable,
        errors: error.errors,
      },
      {
        compact: false,
        depth: Infinity,
      },
    )
    .replaceAll("[Object: null prototype]", "");

  const readable = new Error(message, { cause: error.cause });
  readable.name = error.name;
  const frames = String(error.stack ?? "")
    .split("\n")
    .filter((line) => line.trimStart().startsWith("at "));
  readable.stack = [`${readable.name}: ${message}`, ...frames].join("\n");
  return readable;
}

// FrameworkError extends AggregateError, and Vitest 4 reports a thrown AggregateError
// only through its `errors` array, which hides the error itself (and reports nothing
// at all when the array is empty). Rethrow it as a plain Error with the details inlined.
const wrappedFns = new WeakSet<object>();
beforeEach(({ task }) => {
  const fn = getFn(task);
  if (!fn || wrappedFns.has(fn)) {
    return;
  }
  const wrapped = async () => {
    try {
      return await fn();
    } catch (error) {
      throw isFrameworkErrorLike(error) ? toReadableError(error) : error;
    }
  };
  wrappedFns.add(wrapped);
  setFn(task, wrapped);
});

expect.addSnapshotSerializer({
  serialize(val: FrameworkError): string {
    return val.explain();
  },
  test(val): boolean {
    return val && val instanceof FrameworkError;
  },
});
