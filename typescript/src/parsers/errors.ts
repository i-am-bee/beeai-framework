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
