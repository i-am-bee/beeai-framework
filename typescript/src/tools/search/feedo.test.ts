/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FeedoSearchTool } from "./feedo.js";
import { expect } from "vitest";

describe("FeedoSearchTool", () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = process.env;
    process.env = { ...originalEnv, FEEDO_USAGE_KEY: "dummy_key" };
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  it("should initialize with usage key from env", () => {
    const tool = new FeedoSearchTool();
    expect(tool.options.usageKey).toBe("dummy_key");
  });

  it("should throw if no usage key is provided", () => {
    process.env.FEEDO_USAGE_KEY = "";
    expect(() => new FeedoSearchTool()).toThrowError(/FEEDO_USAGE_KEY is required/);
  });

  describe("run", () => {
    // Note: To fully unit test without network, we'd mock the dynamic import of feedo-protocol-sdk,
    // but vitest module mocking for dynamic imports is slightly complex. 
    // We cover basic schema validation here.
    it("should validate input schema for add", async () => {
      const tool = new FeedoSearchTool();
      const schema = await tool.inputSchema();
      const result = schema.safeParse({ action: "add", query: "test" });
      expect(result.success).toBe(true);
    });

    it("should fail validation for unknown action", async () => {
      const tool = new FeedoSearchTool();
      const schema = await tool.inputSchema();
      const result = schema.safeParse({ action: "unknown", query: "test" });
      expect(result.success).toBe(false);
    });
  });
});
