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
import { ChatModelParameters } from "@/backend/chat.js";

type AmazonBedrockParameters = Parameters<AmazonBedrockProvider["languageModel"]>;
export type AmazonBedrockChatModelId = NonNullable<AmazonBedrockParameters[0]>;

export class AmazonBedrockChatModel extends VercelChatModel {
  constructor(
    modelId: AmazonBedrockChatModelId = getEnv("AWS_CHAT_MODEL", "meta.llama3-70b-instruct-v1:0"),
    parameters: ChatModelParameters = {},
    client?: AmazonBedrockClient | AmazonBedrockClientSettings,
  ) {
    const model = AmazonBedrockClient.ensure(client).instance.languageModel(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }
}
