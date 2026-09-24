/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { XaiClient, type XaiClientSettings } from "@/adapters/xai/backend/client.js";
import { getEnv } from "@/internals/env.js";
import type { XaiProvider } from "@ai-sdk/xai";
import { ChatModelParameters } from "@/backend/chat.js";

type XaiParameters = Parameters<XaiProvider["languageModel"]>;
export type XAIChatModelId = NonNullable<XaiParameters[0]>;

export class XAIChatModel extends VercelChatModel {
  constructor(
    modelId: XAIChatModelId = getEnv("XAI_CHAT_MODEL", "grok-3-mini"),
    parameters: ChatModelParameters = {},
    client?: XaiClientSettings | XaiClient,
  ) {
    const model = XaiClient.ensure(client).instance.languageModel(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  static {
    XAIChatModel.register();
  }
}
