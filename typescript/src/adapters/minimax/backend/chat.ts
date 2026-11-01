/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from "@ai-sdk/openai";
import { MiniMaxClient, MiniMaxClientSettings } from "@/adapters/minimax/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelInput, ChatModelParameters } from "@/backend/chat.js";
import { generateText } from "ai";

type MiniMaxParameters = Parameters<OpenAIProvider["chat"]>;
export type MiniMaxChatModelId = NonNullable<MiniMaxParameters[0]>;

export interface MiniMaxChatModelParameters extends ChatModelParameters {
  // The AI SDK's OpenAI provider rejects MiniMax's "max" level.
  reasoningEffort?: "low" | "medium" | "high" | "xhigh";
}

export class MiniMaxChatModel extends VercelChatModel {
  declare public parameters: MiniMaxChatModelParameters;

  constructor(
    modelId: MiniMaxChatModelId = getEnv("MINIMAX_CHAT_MODEL", "MiniMax-M3"),
    parameters: MiniMaxChatModelParameters = {},
    client?: MiniMaxClient | MiniMaxClientSettings,
  ) {
    const model = MiniMaxClient.ensure(client).instance.chat(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  protected async transformInput(
    input: ChatModelInput,
  ): Promise<Parameters<typeof generateText<Record<string, any>>>[0]> {
    const transformed = (await super.transformInput(input)) as Parameters<
      typeof generateText<Record<string, any>>
    >[0] &
      Pick<MiniMaxChatModelParameters, "reasoningEffort">;
    const { reasoningEffort, providerOptions, ...rest } = transformed;
    if (reasoningEffort === undefined) {
      return transformed;
    }
    return {
      ...rest,
      providerOptions: {
        ...providerOptions,
        openai: { ...providerOptions?.openai, reasoningEffort },
      },
    };
  }

  static {
    this.register();
  }
}
