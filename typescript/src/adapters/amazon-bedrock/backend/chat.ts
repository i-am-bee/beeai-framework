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

import {
  AmazonBedrockClient,
  AmazonBedrockClientSettings,
} from "@/adapters/amazon-bedrock/backend/client.js";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { getEnv } from "@/internals/env.js";
import { AmazonBedrockProvider } from "@ai-sdk/amazon-bedrock";

type AmazonBedrockParameters = Parameters<AmazonBedrockProvider["languageModel"]>;
export type AmazonBedrockChatModelId = NonNullable<AmazonBedrockParameters[0]>;
export type AmazonBedrockChatModelSettings = NonNullable<AmazonBedrockParameters[1]>;

export class AmazonBedrockChatModel extends VercelChatModel {
  constructor(
    modelId: AmazonBedrockChatModelId = getEnv("AWS_CHAT_MODEL", "meta.llama3-70b-instruct-v1:0"),
    settings: AmazonBedrockChatModelSettings = {},
    client?: AmazonBedrockClient | AmazonBedrockClientSettings,
  ) {
    const model = AmazonBedrockClient.ensure(client).instance.languageModel(modelId, settings);
    super(model);
  }
}
