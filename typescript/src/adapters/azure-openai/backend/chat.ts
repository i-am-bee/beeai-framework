/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import type {
  AzureOpenAIProvider as VercelAzureOpenAIProvider,
  AzureOpenAIProviderSettings as VercelAzureOpenAIProviderSettings,
} from "@ai-sdk/azure";
import { AzureOpenAIClient } from "@/adapters/azure-openai/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { ChatModelParameters } from "@/backend/chat.js";

type AzureOpenAIParameters = Parameters<VercelAzureOpenAIProvider["languageModel"]>;
export type AzureOpenAIChatModelId = NonNullable<AzureOpenAIParameters[0]>;

export class AzureOpenAIChatModel extends VercelChatModel {
  constructor(
    modelId: AzureOpenAIChatModelId = getEnv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o"),
    parameters: ChatModelParameters = {},
    client?: VercelAzureOpenAIProviderSettings | AzureOpenAIClient,
  ) {
    const model = AzureOpenAIClient.ensure(client).instance.chat(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }
}
