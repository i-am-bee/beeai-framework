/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from "@ai-sdk/openai";
import { MiniMaxClient, MiniMaxClientSettings } from "@/adapters/minimax/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelParameters } from "@/backend/chat.js";

type MiniMaxParameters = Parameters<OpenAIProvider["chat"]>;
export type MiniMaxChatModelId = NonNullable<MiniMaxParameters[0]>;

export class MiniMaxChatModel extends VercelChatModel {
  constructor(
    modelId: MiniMaxChatModelId = getEnv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7"),
    parameters: ChatModelParameters = {},
    client?: MiniMaxClient | MiniMaxClientSettings,
  ) {
    const model = MiniMaxClient.ensure(client).instance.chat(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  static {
    this.register();
  }
}
