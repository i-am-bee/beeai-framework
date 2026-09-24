/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError, FrameworkErrorOptions } from "@/errors.js";
import { ValueOf } from "@/internals/types.js";

interface Context {
  lines: string;
  excludedLines: string;
  finalState: Record<string, any>;
  partialState: Record<string, any>;
}

export class LinePrefixParserError extends FrameworkError {
  isFatal = true;
  isRetryable = true;
  readonly context: Context;
  readonly reason: ValueOf<typeof LinePrefixParserError.Reason>;

  static Reason = {
    NoDataReceived: "NoDataReceived",
    InvalidTransition: "InvalidTransition",
    NotStartNode: "NotStartNode",
    NotEndNode: "NotEndNode",
    AlreadyCompleted: "AlreadyCompleted",
    InvalidSchema: "InvalidSchema",
  } as const;

  constructor(
    message: string,
    errors: Error[],
    options: FrameworkErrorOptions & {
      context: Context;
      reason: ValueOf<typeof LinePrefixParserError.Reason>;
    },
  ) {
    super(message, errors, options);

    this.context = options.context;
    this.reason = options.reason;
  }
}
