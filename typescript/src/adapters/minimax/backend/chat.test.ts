/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MiniMaxChatModel } from "@/adapters/minimax/backend/chat.js";
import { MiniMaxClientSettings } from "@/adapters/minimax/backend/client.js";
import { ChatModel } from "@/backend/chat.js";
import { UserMessage } from "@/backend/message.js";

// Exercise the real AI SDK and framework request path, replacing only HTTP I/O.
const requests: { url: string; body: Record<string, any> }[] = [];
const mockFetch: typeof fetch = async (url, init) => {
  const body = JSON.parse(String(init?.body));
  requests.push({ url: String(url), body });
  const response = {
    id: "chatcmpl-test",
    object: "chat.completion",
    created: 1,
    model: body.model,
    choices: [
      { index: 0, message: { role: "assistant", content: "Hello" }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  if (body.stream) {
    const chunk = {
      ...response,
      object: "chat.completion.chunk",
      choices: [
        { index: 0, delta: { role: "assistant", content: "Hello" }, finish_reason: "stop" },
      ],
    };
    return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, {
      headers: { "content-type": "text/event-stream" },
    });
  }
  return Response.json(response);
};

beforeEach(() => {
  requests.length = 0;
  for (const key of [
    "MINIMAX_CHAT_MODEL",
    "MINIMAX_API_BASE",
    "MINIMAX_API_REGION",
    "MINIMAX_API_HEADERS",
  ]) {
    vi.stubEnv(key, undefined);
  }
  vi.stubEnv("MINIMAX_API_KEY", "test-key");
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe.each(["global", "cn"] as const)("MiniMax %s requests", (region) => {
  it.each([undefined, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])(
    "preserves model selection %s",
    async (modelId) => {
      for (const stream of [false, true]) {
        requests.length = 0;
        const model = new MiniMaxChatModel(modelId, {}, { region });
        const result = await model.create({ messages: [new UserMessage("Hello")], stream });
        expect(result.getTextContent()).toBe("Hello");
        expect(requests).toHaveLength(1);
        const { url, body } = requests[0];
        expect(url).toBe(
          `https://${region === "cn" ? "api.minimaxi.com" : "api.minimax.io"}/v1/chat/completions`,
        );
        expect(body.model).toBe(modelId ?? "MiniMax-M3");
        expect(body.messages).toEqual([{ role: "user", content: "Hello" }]);
        expect(Boolean(body.stream)).toBe(stream);
        for (const key of ["thinking", "reasoning_effort", "service_tier", "reasoning_split"]) {
          expect(body).not.toHaveProperty(key);
        }
      }
    },
  );
});

it("uses M3 through the provider factory", async () => {
  const model = await ChatModel.fromName("minimax");
  await model.create({ messages: [new UserMessage("Hello")] });
  expect(requests[0].body.model).toBe("MiniMax-M3");
});

it.each(["MiniMax-M3.1-Flash", "custom-model"])(
  "passes reasoning effort through for %s",
  async (modelId) => {
    const model = new MiniMaxChatModel(modelId, { reasoningEffort: "high" });
    await model.create({ messages: [new UserMessage("Hello")] });
    expect(requests[0].body.reasoning_effort).toBe("high");
  },
);

it.each(["MiniMax-M3", "MiniMax-M2.7-highspeed", "custom-model"])(
  "preserves the saved model %s and custom endpoint",
  async (modelId) => {
    vi.stubEnv("MINIMAX_CHAT_MODEL", modelId);
    vi.stubEnv("MINIMAX_API_REGION", "cn");
    vi.stubEnv("MINIMAX_API_BASE", "https://proxy.example.com/v1");
    const model = await ChatModel.fromName("minimax");
    await model.create({ messages: [new UserMessage("Hello")] });
    expect(requests[0].url).toBe("https://proxy.example.com/v1/chat/completions");
    expect(requests[0].body.model).toBe(modelId);
  },
);

it("preserves image content", async () => {
  const model = new MiniMaxChatModel();
  await model.create({
    messages: [
      new UserMessage([
        { type: "text", text: "Describe this image" },
        { type: "image", image: new URL("https://example.com/image.png") },
      ]),
    ],
  });
  expect(requests[0].body.messages[0].content).toEqual([
    { type: "text", text: "Describe this image" },
    { type: "image_url", image_url: { url: "https://example.com/image.png" } },
  ]);
});

it("preserves explicit model and endpoint overrides", async () => {
  vi.stubEnv("MINIMAX_CHAT_MODEL", "saved-model");
  vi.stubEnv("MINIMAX_API_BASE", "https://saved.example.com/v1");
  const settings: MiniMaxClientSettings = {
    baseURL: "https://custom.example.com/v1",
    fetch: mockFetch,
  };
  const model = new MiniMaxChatModel("custom-model", {}, settings);
  await model.create({ messages: [new UserMessage("Hello")] });
  expect(requests[0].url).toBe("https://custom.example.com/v1/chat/completions");
  expect(requests[0].body.model).toBe("custom-model");
});
