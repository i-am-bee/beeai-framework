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
import { AnthropicClient, AnthropicClientSettings } from "@/adapters/anthropic/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { AnthropicProvider } from "@ai-sdk/anthropic";

type AnthropicParameters = Parameters<AnthropicProvider["languageModel"]>;
export type AnthropicChatModelId = NonNullable<AnthropicParameters[0]>;
export type AnthropicChatModelSettings = NonNullable<AnthropicParameters[1]>;

export class AnthropicChatModel extends VercelChatModel {
  constructor(
    modelId: AnthropicChatModelId = getEnv("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest"),
    settings: AnthropicChatModelSettings = {},
    client?: AnthropicClientSettings | AnthropicClient,
  ) {
    const model = AnthropicClient.ensure(client).instance.languageModel(modelId, settings);
    super(model);
  }

  static {
    this.register();
  }
}
