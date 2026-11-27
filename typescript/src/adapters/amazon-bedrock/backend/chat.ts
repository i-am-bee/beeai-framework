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
import { generateText } from "ai";
import { ChatModelInput, ChatModelParameters } from "@/backend/chat.js";

type AmazonBedrockParameters = Parameters<AmazonBedrockProvider["languageModel"]>;
export type AmazonBedrockChatModelId = NonNullable<AmazonBedrockParameters[0]>;

// Same split as @ai-sdk/amazon-bedrock: OpenAI models on Bedrock other than gpt-oss (GPT-5.x, GPT-6)
// reject the `temperature` field on Converse.
const OPENAI_WITHOUT_TEMPERATURE = /^(?:[^.]+\.)?openai\.(?!gpt-oss-)/;
// Claude Opus 4.7+, Sonnet 5+, Haiku 5+, Fable and Mythos reject any `temperature` other than 1.
// Same set as `_CLAUDE_WITHOUT_SAMPLING_PARAMS` in the Python package.
const CLAUDE_WITHOUT_TEMPERATURE =
  /claude-(?:fable|mythos|opus-4-[7-9]|(?:opus|sonnet|haiku)-(?:[5-9]|\d\d))/;

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

  protected async transformInput(
    input: ChatModelInput,
  ): Promise<Parameters<typeof generateText<Record<string, any>>>[0]> {
    const transformed = await super.transformInput(input);
    // Drop the framework's default `temperature: 0`; a temperature set by the user is still sent.
    const rejectsTemperature =
      OPENAI_WITHOUT_TEMPERATURE.test(this.modelId) ||
      CLAUDE_WITHOUT_TEMPERATURE.test(this.modelId);
    return rejectsTemperature ? { temperature: undefined, ...transformed } : transformed;
  }

  static {
    this.register();
  }
}

// `ChatModel.fromName()` looks up `BedrockChatModel`, like it finds `BedrockEmbeddingModel`.
export { AmazonBedrockChatModel as BedrockChatModel };
