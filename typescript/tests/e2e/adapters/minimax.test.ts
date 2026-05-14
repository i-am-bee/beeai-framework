/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { BackendProviders } from "@/backend/constants.js";
import { MiniMaxChatModel } from "@/adapters/minimax/backend/chat.js";
import { MiniMaxClient } from "@/adapters/minimax/backend/client.js";

describe("MiniMax Provider Registration", () => {
  it("should be registered in BackendProviders", () => {
    expect(BackendProviders.MiniMax).toBeDefined();
    expect(BackendProviders.MiniMax.name).toBe("MiniMax");
    expect(BackendProviders.MiniMax.module).toBe("minimax");
    expect(BackendProviders.MiniMax.aliases).toContain("minimax");
  });
});

describe("MiniMaxClient", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("should create client with explicit settings", () => {
    const client = new MiniMaxClient({
      apiKey: "test-key",
      baseURL: "https://api.minimax.io/v1",
    });
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });

  it("should create client from env vars", () => {
    process.env.MINIMAX_API_KEY = "test-key-from-env";
    const client = new MiniMaxClient({});
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });
});

describe("MiniMaxChatModel", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    process.env.MINIMAX_API_KEY = "test-api-key";
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("should instantiate with default model", () => {
    const model = new MiniMaxChatModel();
    expect(model).toBeInstanceOf(MiniMaxChatModel);
    expect(model.modelId).toBe("MiniMax-M2.7");
  });

  it("should instantiate with custom model id", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.5");
    expect(model).toBeInstanceOf(MiniMaxChatModel);
    expect(model.modelId).toBe("MiniMax-M2.5");
  });

  it("should accept highspeed model", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.7-highspeed");
    expect(model).toBeInstanceOf(MiniMaxChatModel);
    expect(model.modelId).toBe("MiniMax-M2.7-highspeed");
  });

  it("should use env var for model id", () => {
    process.env.MINIMAX_CHAT_MODEL = "MiniMax-M2.5-highspeed";
    const model = new MiniMaxChatModel();
    expect(model.modelId).toBe("MiniMax-M2.5-highspeed");
  });

  it("should accept custom parameters", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.7", { temperature: 0.5 });
    expect(model).toBeInstanceOf(MiniMaxChatModel);
  });

  it("should accept custom client settings", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.7", {}, {
      apiKey: "custom-key",
      baseURL: "https://proxy.example.com/v1",
    });
    expect(model).toBeInstanceOf(MiniMaxChatModel);
  });
});
