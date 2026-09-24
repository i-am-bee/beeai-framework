/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AmazonBedrockChatModel } from "@/adapters/amazon-bedrock/backend/chat.js";
import { ChatModelParameters } from "@/backend/chat.js";
import { UserMessage } from "@/backend/message.js";

async function sentInferenceConfig(modelId: string, parameters: ChatModelParameters = {}) {
  let body: Record<string, any> = {};
  const llm = new AmazonBedrockChatModel(modelId, parameters, {
    region: "us-east-1",
    apiKey: "test",
    fetch: async (_url, init) => {
      body = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          output: { message: { role: "assistant", content: [{ text: "pong" }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        }),
        { headers: { "content-type": "application/json" } },
      );
    },
  });
  await llm.create({ messages: [new UserMessage("ping")] });
  return body.inferenceConfig ?? {};
}

describe("AmazonBedrockChatModel", () => {
  it.each([
    "us.openai.gpt-6-sol",
    "global.openai.gpt-6-luna",
    "us.openai.gpt-6-astra",
    "us.openai.gpt-5.6-sol",
  ])("does not send the default temperature to %s", async (modelId) => {
    expect(await sentInferenceConfig(modelId)).not.toHaveProperty("temperature");
  });

  it.each(["openai.gpt-oss-120b-1:0", "meta.llama3-70b-instruct-v1:0"])(
    "keeps the default temperature for %s",
    async (modelId) => {
      expect(await sentInferenceConfig(modelId)).toEqual({ temperature: 0 });
    },
  );

  it("still sends a temperature set by the user", async () => {
    expect(await sentInferenceConfig("us.openai.gpt-6-sol", { temperature: 0.5 })).toEqual({
      temperature: 0.5,
    });
  });
});
