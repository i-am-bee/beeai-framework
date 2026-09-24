/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { GroqClient, GroqClientSettings } from "@/adapters/groq/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { GroqProvider } from "@ai-sdk/groq";
import { ChatModelInput, ChatModelParameters } from "@/backend/chat.js";
import type { GetRunContext } from "@/context.js";
import { APICallError } from "ai";
import { ChatModelError, ChatModelToolCallError } from "@/backend/errors.js";
import { parseBrokenJson } from "@/internals/helpers/schema.js";
import { isPlainObject } from "remeda";

type GroqParameters = Parameters<GroqProvider["languageModel"]>;
export type GroqChatModelId = NonNullable<GroqParameters[0]>;

export class GroqChatModel extends VercelChatModel {
  constructor(
    modelId: GroqChatModelId = getEnv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
    parameters: ChatModelParameters = {},
    client?: GroqClientSettings | GroqClient,
  ) {
    const model = GroqClient.ensure(client).instance.languageModel(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  static {
    this.register();
  }

  async *_createStream(input: ChatModelInput, run: GetRunContext<this>) {
    try {
      for await (const chunk of super._createStream(input, run)) {
        yield chunk;
      }
    } catch (error) {
      this.handleError(input, error);
    }
  }

  protected async _create(input: ChatModelInput, run: GetRunContext<this>) {
    return await super._create(input, run).catch((e) => this.handleError(input, e));
  }

  protected handleError(input: ChatModelInput, error: Error): never {
    const matchedErrorMessages = [
      "model did not call a tool",
      "tool call validation failed",
      "tool choice is required",
      "Parsing failed",
    ];

    const cause: Error | Record<string, any> =
      error instanceof ChatModelError ? error.getCause() : error;
    const responseBodyRaw = APICallError.isInstance(cause)
      ? cause.responseBody
      : isPlainObject(cause)
        ? JSON.stringify(cause)
        : undefined;

    if (
      responseBodyRaw &&
      matchedErrorMessages.some((message) => responseBodyRaw.includes(message))
    ) {
      const responseBody = parseBrokenJson(responseBodyRaw, { pair: ["{", "}"] });
      if (!responseBody) {
        throw cause;
      }

      const tools = (input.tools || []).map((t) => t.name).join(", ");
      throw new ChatModelToolCallError(
        responseBody?.error?.message || responseBody?.message || String(responseBody),
        [],
        {
          generatedContent: responseBody?.error?.failed_generation || "empty",
          generatedError: `Invalid response. Use one of the following tools: ${tools}. `,
        },
      );
    }
    throw cause;
  }
}
