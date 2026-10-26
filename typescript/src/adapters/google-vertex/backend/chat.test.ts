/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleVertexChatModel } from "@/adapters/google-vertex/backend/chat.js";

describe("GoogleVertexChatModel", () => {
  const original = process.env.GOOGLE_VERTEX_CHAT_MODEL;

  afterEach(() => {
    vi.unstubAllEnvs();
    if (original === undefined) {
      delete process.env.GOOGLE_VERTEX_CHAT_MODEL;
    } else {
      process.env.GOOGLE_VERTEX_CHAT_MODEL = original;
    }
  });

  it("defaults to a model that has not been retired", () => {
    delete process.env.GOOGLE_VERTEX_CHAT_MODEL;
    const llm = new GoogleVertexChatModel(
      undefined,
      {},
      { project: "test", location: "us-central1" },
    );
    expect(llm.modelId).toBe("gemini-3.8-flash");
  });

  it("prefers GOOGLE_VERTEX_CHAT_MODEL over the default", () => {
    vi.stubEnv("GOOGLE_VERTEX_CHAT_MODEL", "gemini-3.6-flash");
    const llm = new GoogleVertexChatModel(
      undefined,
      {},
      { project: "test", location: "us-central1" },
    );
    expect(llm.modelId).toBe("gemini-3.6-flash");
  });
});
