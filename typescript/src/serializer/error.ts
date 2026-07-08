/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";

export class SerializerError extends FrameworkError {
  constructor(message: string, errors?: Error[]) {
    super(message, errors, {
      isFatal: true,
      isRetryable: false,
    });
  }
}
