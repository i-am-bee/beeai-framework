/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
