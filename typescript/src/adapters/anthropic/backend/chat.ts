/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { AnthropicClient, AnthropicClientSettings } from "@/adapters/anthropic/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { AnthropicProvider } from "@ai-sdk/anthropic";
import { ChatModelParameters } from "@/backend/chat.js";

type AnthropicParameters = Parameters<AnthropicProvider["languageModel"]>;
export type AnthropicChatModelId = NonNullable<AnthropicParameters[0]>;

export class AnthropicChatModel extends VercelChatModel {
  constructor(
    modelId: AnthropicChatModelId = getEnv("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest"),
    parameters: ChatModelParameters = {},
    client?: AnthropicClientSettings | AnthropicClient,
  ) {
    const model = AnthropicClient.ensure(client).instance.languageModel(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  static {
    this.register();
  }
}
