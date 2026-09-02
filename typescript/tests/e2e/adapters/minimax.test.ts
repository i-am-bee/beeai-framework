/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { BackendProviders } from "@/backend/constants.js";
import { MiniMaxChatModel } from "@/adapters/minimax/backend/chat.js";
import {
  MiniMaxClient,
  MINIMAX_API_BASE,
  MINIMAX_API_BASE_CN,
  resolveMiniMaxBaseURL,
} from "@/adapters/minimax/backend/client.js";

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
    expect(model.modelId).toBe("MiniMax-M3");
  });

  it("should instantiate with custom model id", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.7");
    expect(model).toBeInstanceOf(MiniMaxChatModel);
    expect(model.modelId).toBe("MiniMax-M2.7");
  });

  it("should accept highspeed model", () => {
    const model = new MiniMaxChatModel("MiniMax-M2.7-highspeed");
    expect(model).toBeInstanceOf(MiniMaxChatModel);
    expect(model.modelId).toBe("MiniMax-M2.7-highspeed");
  });

  it("should use env var for model id", () => {
    process.env.MINIMAX_CHAT_MODEL = "MiniMax-M2.7-highspeed";
    const model = new MiniMaxChatModel();
    expect(model.modelId).toBe("MiniMax-M2.7-highspeed");
  });

  it("should accept custom parameters", () => {
    const model = new MiniMaxChatModel("MiniMax-M3", { temperature: 0.5 });
    expect(model).toBeInstanceOf(MiniMaxChatModel);
  });

  it("should accept custom client settings", () => {
    const model = new MiniMaxChatModel(
      "MiniMax-M3",
      {},
      {
        apiKey: "custom-key",
        baseURL: "https://proxy.example.com/v1",
      },
    );
    expect(model).toBeInstanceOf(MiniMaxChatModel);
  });
});

describe("MiniMax regional endpoints", () => {
  it("should expose distinct global and CN endpoints", () => {
    expect(MINIMAX_API_BASE).toBe("https://api.minimax.io/v1");
    expect(MINIMAX_API_BASE_CN).toBe("https://api.minimaxi.com/v1");
    expect(MINIMAX_API_BASE).not.toBe(MINIMAX_API_BASE_CN);
  });

  it("should resolve the global region", () => {
    expect(resolveMiniMaxBaseURL("global")).toBe(MINIMAX_API_BASE);
  });

  it("should resolve the CN region", () => {
    expect(resolveMiniMaxBaseURL("cn")).toBe(MINIMAX_API_BASE_CN);
  });

  it.each(["global_en", "cn_zh"])("should reject for unsupported alias %s", (region) => {
    expect(() => resolveMiniMaxBaseURL(region)).toThrowError(/Unknown MiniMax region/);
  });

  it("should be case-insensitive and trim whitespace", () => {
    expect(resolveMiniMaxBaseURL("  CN  ")).toBe(MINIMAX_API_BASE_CN);
  });

  it("should default to the global endpoint when region is missing", () => {
    expect(resolveMiniMaxBaseURL()).toBe(MINIMAX_API_BASE);
    expect(resolveMiniMaxBaseURL("")).toBe(MINIMAX_API_BASE);
  });

  it("should throw for an unknown region", () => {
    expect(() => resolveMiniMaxBaseURL("mars")).toThrowError(/Unknown MiniMax region/);
  });

  it("should build a client for the CN region without throwing", () => {
    const client = new MiniMaxClient({ apiKey: "test-key", region: "cn" });
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });

  it("should ignore an invalid region when baseURL is explicit", () => {
    const client = new MiniMaxClient({
      apiKey: "test-key",
      baseURL: "https://proxy.example.com/v1",
      region: "mars" as never,
    });
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });
});
