/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatModel } from "@/backend/chat.js";
import { AmazonBedrockChatModel } from "@/adapters/amazon-bedrock/backend/chat.js";

describe("ChatModel.fromName", () => {
  it("creates an Amazon Bedrock chat model", async () => {
    const model = await ChatModel.fromName("amazon-bedrock:us.openai.gpt-6-sol");
    expect(model).toBeInstanceOf(AmazonBedrockChatModel);
    expect(model.modelId).toBe("us.openai.gpt-6-sol");
  });

  it("clones an Amazon Bedrock chat model", async () => {
    const model = new AmazonBedrockChatModel("us.openai.gpt-6-sol");
    const clone = await model.clone();
    expect(clone).toBeInstanceOf(AmazonBedrockChatModel);
    expect(clone.modelId).toBe("us.openai.gpt-6-sol");
  });
});
