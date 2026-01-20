/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";
import type { ChatModelOutput } from "@/backend/chat.js";

export class BackendError extends FrameworkError {}

export class ChatModelError extends BackendError {}

export class EmptyChatModelResponseError extends ChatModelError {
  constructor(message = "Chat Model produced an empty response!") {
    super(message, [], {
      isFatal: true,
      isRetryable: true,
    });
  }
}

export class ChatModelToolCallError extends ChatModelError {
  constructor(
    message: string,
    errors: Error[] = [],
    public readonly data: {
      generatedContent: string;
      generatedError: string;
      response?: ChatModelOutput;
    },
  ) {
    super(message, errors, {
      isFatal: true,
      isRetryable: true,
    });
  }
}

export class EmbeddingModelError extends BackendError {}
