/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from "@ai-sdk/openai";
import {
  OrcaRouterClient,
  OrcaRouterClientSettings,
} from "@/adapters/orcarouter/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelParameters } from "@/backend/chat.js";

type OrcaRouterParameters = Parameters<OpenAIProvider["chat"]>;
export type OrcaRouterChatModelId = NonNullable<OrcaRouterParameters[0]>;

export class OrcaRouterChatModel extends VercelChatModel {
  constructor(
    modelId: OrcaRouterChatModelId = getEnv("ORCAROUTER_CHAT_MODEL", "orcarouter/auto"),
    parameters: ChatModelParameters = {},
    client?: OrcaRouterClient | OrcaRouterClientSettings,
  ) {
    const model = OrcaRouterClient.ensure(client).instance.chat(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  static {
    this.register();
  }
}
