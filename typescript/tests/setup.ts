/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import dotenv from "dotenv";
import util from "util";
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

afterEach(() => {
  onTestFailed((testCase) => {
    const errors = testCase.errors ?? [];
    for (const error of errors) {
      if (isFrameworkErrorLike(error)) {
        error.message = util
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
      }
    }
  });
});

expect.addSnapshotSerializer({
  serialize(val: FrameworkError): string {
    return val.explain();
  },
  test(val): boolean {
    return val && val instanceof FrameworkError;
  },
});
