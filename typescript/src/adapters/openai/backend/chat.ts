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

import { OpenAIProvider } from "@ai-sdk/openai";
import { OpenAIClient, OpenAIClientSettings } from "@/adapters/openai/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";

type OpenAIParameters = Parameters<OpenAIProvider["chat"]>;
export type OpenAIChatModelId = NonNullable<OpenAIParameters[0]>;
export type OpenAIChatModelSettings = NonNullable<OpenAIParameters[1]>;

export class OpenAIChatModel extends VercelChatModel {
  constructor(
    modelId: OpenAIChatModelId = getEnv("OPENAI_CHAT_MODEL", "gpt-4o"),
    settings: OpenAIChatModelSettings = {},
    client?: OpenAIClient | OpenAIClientSettings,
  ) {
    const model = OpenAIClient.ensure(client).instance.chat(modelId, settings);
    super(model);
  }
}
