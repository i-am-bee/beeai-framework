/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { GroqClient, GroqClientSettings } from "@/adapters/groq/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { GroqProvider } from "@ai-sdk/groq";
import { ChatModelInput, ChatModelParameters } from "@/backend/chat.js";
import { GetRunContext } from "@/context.js";
import { APICallError } from "ai";
import { ChatModelToolCallError } from "@/backend/errors.js";
import { parseBrokenJson } from "@/internals/helpers/schema.js";

type GroqParameters = Parameters<GroqProvider["languageModel"]>;
export type GroqChatModelId = NonNullable<GroqParameters[0]>;

export class GroqChatModel extends VercelChatModel {
  constructor(
    modelId: GroqChatModelId = getEnv("GROQ_CHAT_MODEL", "gemma2-9b-it"),
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

  protected async _create(input: ChatModelInput, run: GetRunContext<this>) {
    try {
      return await super._create(input, run);
    } catch (e) {
      if (
        APICallError.isInstance(e) &&
        (e.responseBody?.includes("model did not call a tool") ||
          e.responseBody?.includes("tool call validation failed"))
      ) {
        const responseBody = parseBrokenJson(e.responseBody, { pair: ["{", "}"] });
        if (!responseBody) {
          throw e;
        }

        const {
          error: { failed_generation: failedGeneration, message: errorMessage },
        } = responseBody;
        const tools = (input.tools || []).map((t) => t.name).join(", ");

        throw new ChatModelToolCallError(errorMessage, [], {
          generatedContent: failedGeneration || "empty",
          generatedError: `Invalid response. Use one of the following tools: ${tools}. `,
        });
      }
      throw e;
    }
  }
}
