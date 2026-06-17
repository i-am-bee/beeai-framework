/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ResponsesRequestInputMessage } from "./responses_types.js";
import { Message, UserMessage, AssistantMessage, SystemMessage } from "@/backend/message.js";

export function openaiInputToBeeAIMessage(message: ResponsesRequestInputMessage): Message {
  switch (message.role) {
    case "user":
      return new UserMessage(message.content ?? "");
    case "system":
    case "developer":
      return new SystemMessage(message.content ?? "");
    case "assistant":
      return new AssistantMessage(message.content ?? "");
    default:
      throw new Error(`Invalid role: ${message.role}`);
  }
}
