/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatModelInput, ChatModelParameters } from "@/backend/chat.js";
import { UserMessage } from "@/backend/message.js";
import { z } from "zod";
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

describe("AnthropicChatModel sampling parameters", () => {
  async function sentBody(
    parameters: ChatModelParameters = {},
    input: Partial<ChatModelInput> = {},
    mode: "text" | "stream" | "structure" = "text",
  ) {
    let body: Record<string, any> = {};
    const llm = new AnthropicChatModel("claude-sonnet-4-5", parameters, {
      apiKey: "test",
      fetch: async (_url, init) => {
        body = JSON.parse(String(init?.body));
        const message = {
          id: "msg_test",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-4-5",
          content: [{ type: "text", text: '{"answer":"ok"}' }],
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
        };
        if (body.stream) {
          const events = [
            { type: "message_start", message: { ...message, content: [], stop_reason: null } },
            { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
            {
              type: "content_block_delta",
              index: 0,
              delta: { type: "text_delta", text: "ok" },
            },
            { type: "content_block_stop", index: 0 },
            {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: 1 },
            },
            { type: "message_stop" },
          ];
          return new Response(
            events
              .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
              .join(""),
            { headers: { "content-type": "text/event-stream" } },
          );
        }
        return Response.json(message);
      },
    });
    const request = { messages: [new UserMessage("hi")], ...input };
    if (mode === "structure") {
      await llm.createStructure({ ...request, schema: z.object({ answer: z.string() }) });
    } else {
      await llm.create({ ...request, stream: mode === "stream" });
    }
    return body;
  }

  it.each(["text", "stream", "structure"] as const)(
    "sends top_p without the implicit temperature for %s requests",
    async (mode) => {
      const body = await sentBody({}, { topP: 0.8 }, mode);
      expect(body.top_p).toBe(0.8);
      expect(body).not.toHaveProperty("temperature");
    },
  );

  it("uses a model-level topP", async () => {
    const body = await sentBody({ topP: 0.7 });
    expect(body.top_p).toBe(0.7);
    expect(body).not.toHaveProperty("temperature");
  });

  it("prefers per-call topP over model-level topP", async () => {
    expect((await sentBody({ topP: 0.7 }, { topP: 0.9 })).top_p).toBe(0.9);
  });

  it.each([0, 0.4])("keeps an explicit model-level temperature of %s", async (temperature) => {
    expect((await sentBody({ temperature }, { topP: 0.8 })).temperature).toBe(temperature);
  });

  it.each([0, 0.4])("keeps an explicit per-call temperature of %s", async (temperature) => {
    expect((await sentBody({ topP: 0.8 }, { temperature })).temperature).toBe(temperature);
  });

  it("retains the default temperature without topP", async () => {
    expect((await sentBody()).temperature).toBe(0);
  });
});
