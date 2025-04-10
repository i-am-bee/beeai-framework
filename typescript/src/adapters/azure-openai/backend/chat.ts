/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import type {
  AzureOpenAIProvider as VercelAzureOpenAIProvider,
  AzureOpenAIProviderSettings as VercelAzureOpenAIProviderSettings,
} from "@ai-sdk/azure";
import { AzureOpenAIClient } from "@/adapters/azure-openai/backend/client.js";
import { getEnv } from "@/internals/env.js";

type AzureOpenAIParameters = Parameters<VercelAzureOpenAIProvider["languageModel"]>;
export type AzureOpenAIChatModelId = NonNullable<AzureOpenAIParameters[0]>;
export type AzureOpenAIChatModelSettings = NonNullable<AzureOpenAIParameters[1]>;

export class AzureOpenAIChatModel extends VercelChatModel {
  constructor(
    modelId: AzureOpenAIChatModelId = getEnv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o"),
    settings: AzureOpenAIChatModelSettings = {},
    client?: VercelAzureOpenAIProviderSettings | AzureOpenAIClient,
  ) {
    const model = AzureOpenAIClient.ensure(client).instance.chat(modelId, settings);
    super(model);
  }
}
