/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from "@ai-sdk/openai";
import { OpenAIClient, OpenAIClientSettings } from "@/adapters/openai/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelParameters } from "@/backend/chat.js";

type OpenAIParameters = Parameters<OpenAIProvider["chat"]>;
export type OpenAIChatModelId = NonNullable<OpenAIParameters[0]>;

export class OpenAIChatModel extends VercelChatModel {
  constructor(
    modelId: OpenAIChatModelId = getEnv("OPENAI_CHAT_MODEL", "gpt-4o"),
    parameters: ChatModelParameters = {},
    client?: OpenAIClient | OpenAIClientSettings,
  ) {
    const model = OpenAIClient.ensure(client).instance.chat(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }
}
