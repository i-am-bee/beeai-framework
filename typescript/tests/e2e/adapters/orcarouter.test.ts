/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { BackendProviders } from "@/backend/constants.js";
import { OrcaRouterChatModel } from "@/adapters/orcarouter/backend/chat.js";
import { OrcaRouterClient } from "@/adapters/orcarouter/backend/client.js";

describe("OrcaRouter Provider Registration", () => {
  it("should be registered in BackendProviders", () => {
    expect(BackendProviders.OrcaRouter).toBeDefined();
    expect(BackendProviders.OrcaRouter.name).toBe("OrcaRouter");
    expect(BackendProviders.OrcaRouter.module).toBe("orcarouter");
    expect(BackendProviders.OrcaRouter.aliases).toContain("orcarouter");
  });
});

describe("OrcaRouterClient", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("should create client with explicit settings", () => {
    const client = new OrcaRouterClient({
      apiKey: "test-key",
      baseURL: "https://api.orcarouter.ai/v1",
    });
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });

  it("should create client from env vars", () => {
    process.env.ORCAROUTER_API_KEY = "test-key";
    process.env.ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1";
    const client = OrcaRouterClient.ensure();
    expect(client).toBeDefined();
    expect(client.instance).toBeDefined();
  });
});

describe("OrcaRouterChatModel", () => {
  it("should create model instance", () => {
    const model = new OrcaRouterChatModel("orcarouter/auto", {}, { apiKey: "test-key" });
    expect(model).toBeInstanceOf(OrcaRouterChatModel);
    expect(model.modelId).toBe("orcarouter/auto");
  });

  it("should use default model id from env", () => {
    process.env.ORCAROUTER_CHAT_MODEL = "deepseek/deepseek-v4-flash";
    const model = new OrcaRouterChatModel(undefined, {}, { apiKey: "test-key" });
    expect(model.modelId).toBe("deepseek/deepseek-v4-flash");
  });
});
