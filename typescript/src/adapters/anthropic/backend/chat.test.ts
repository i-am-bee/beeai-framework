/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnthropicChatModel } from "@/adapters/anthropic/backend/chat.js";

describe("AnthropicChatModel", () => {
  const original = process.env.ANTHROPIC_CHAT_MODEL;

  afterEach(() => {
    vi.unstubAllEnvs();
    if (original === undefined) {
      delete process.env.ANTHROPIC_CHAT_MODEL;
    } else {
      process.env.ANTHROPIC_CHAT_MODEL = original;
    }
  });

  it("defaults to a model that has not been retired", () => {
    delete process.env.ANTHROPIC_CHAT_MODEL;
    const llm = new AnthropicChatModel(undefined, {}, { apiKey: "test" });
    expect(llm.modelId).toBe("claude-haiku-4-5");
  });

  it("prefers ANTHROPIC_CHAT_MODEL over the default", () => {
    vi.stubEnv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-5");
    const llm = new AnthropicChatModel(undefined, {}, { apiKey: "test" });
    expect(llm.modelId).toBe("claude-sonnet-4-5");
  });
});
