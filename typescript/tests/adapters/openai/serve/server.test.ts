/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { openaiMessageToBeeAIMessage } from "@/adapters/openai/serve/utils.js";
import { openaiInputToBeeAIMessage } from "@/adapters/openai/serve/responses_utils.js";
import { UserMessage, SystemMessage, AssistantMessage, ToolMessage } from "@/backend/message.js";
import { OpenAIServer, OpenAIServerConfig, logger } from "@/adapters/openai/serve/server.js";
import { BaseAgent } from "@/agents/base.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import type { AnyAgent } from "@/agents/types.js";

const INSECURE_WARNING_FRAGMENT = "without authentication";

function hasInsecureWarning(warnings: string[]): boolean {
  return warnings.some((w) => w.includes(INSECURE_WARNING_FRAGMENT));
}

class DummyAgent extends BaseAgent<unknown, unknown> {
  public readonly emitter = new Emitter();
  public memory = new UnconstrainedMemory();

  protected _run(): Promise<unknown> {
    return Promise.resolve("pong");
  }
}

function createDummyAgent(): AnyAgent {
  return new DummyAgent() as unknown as AnyAgent;
}

describe("OpenAIServer utils", () => {
  describe("openaiMessageToBeeAIMessage", () => {
    it("converts user messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "user",
        content: "Hello world",
      });
      expect(msg).toBeInstanceOf(UserMessage);
      expect(msg.text).toBe("Hello world");
    });

    it("converts system messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "system",
        content: "You are a helpful assistant",
      });
      expect(msg).toBeInstanceOf(SystemMessage);
      expect(msg.text).toBe("You are a helpful assistant");
    });

    it("converts assistant messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "assistant",
        content: "Sure, I can help",
      });
      expect(msg).toBeInstanceOf(AssistantMessage);
      expect(msg.text).toBe("Sure, I can help");
    });

    it("converts assistant tool calls correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_123",
            type: "function",
            function: {
              name: "get_weather",
              arguments: '{"location":"New York"}',
            },
          },
        ],
      });
      expect(msg).toBeInstanceOf(AssistantMessage);
      const toolCalls = (msg as AssistantMessage).getToolCalls();
      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0].toolName).toBe("get_weather");
    });

    it("converts tool messages correctly", () => {
      const msg = openaiMessageToBeeAIMessage({
        role: "tool",
        content: "Sunny",
        tool_call_id: "call_123",
        name: "get_weather",
      });
      expect(msg).toBeInstanceOf(ToolMessage);
      const toolResults = (msg as ToolMessage).getToolResults();
      expect(toolResults).toHaveLength(1);
      expect(toolResults[0].toolName).toBe("get_weather");
      expect(toolResults[0].toolCallId).toBe("call_123");
    });
  });
});

describe("Responses API utils", () => {
  describe("openaiInputToBeeAIMessage", () => {
    it("converts user messages correctly", () => {
      const msg = openaiInputToBeeAIMessage({
        role: "user",
        content: "Hello",
        type: "message",
      });
      expect(msg).toBeInstanceOf(UserMessage);
      expect(msg.text).toBe("Hello");
    });

    it("converts system messages correctly", () => {
      const msg = openaiInputToBeeAIMessage({
        role: "system",
        content: "Be helpful",
        type: "message",
      });
      expect(msg).toBeInstanceOf(SystemMessage);
      expect(msg.text).toBe("Be helpful");
    });

    it("converts developer messages to SystemMessage", () => {
      const msg = openaiInputToBeeAIMessage({
        role: "developer",
        content: "Developer instruction",
        type: "message",
      });
      expect(msg).toBeInstanceOf(SystemMessage);
      expect(msg.text).toBe("Developer instruction");
    });

    it("converts assistant messages correctly", () => {
      const msg = openaiInputToBeeAIMessage({
        role: "assistant",
        content: "I can help",
        type: "message",
      });
      expect(msg).toBeInstanceOf(AssistantMessage);
      expect(msg.text).toBe("I can help");
    });

    it("handles null content as empty string", () => {
      const msg = openaiInputToBeeAIMessage({
        role: "user",
        content: null,
        type: "message",
      });
      expect(msg).toBeInstanceOf(UserMessage);
      expect(msg.text).toBe("");
    });

    it("throws on invalid role", () => {
      expect(() =>
        openaiInputToBeeAIMessage({
          role: "invalid" as any,
          content: "test",
          type: "message",
        }),
      ).toThrow("Invalid role: invalid");
    });
  });
});

// Prevent express from actually binding a port. The serve() method calls
// app.listen(port, host, cb); we capture and immediately invoke the callback.
const { mockExpressApp, mockExpressRouter, mockExpressJson, listenMock } = vi.hoisted(() => {
  const listenMock = vi.fn((_port: number, _host: string, cb?: () => void) => {
    cb?.();
    return { on: vi.fn() };
  });
  const mockExpressApp = vi.fn(() => ({
    use: vi.fn(),
    listen: listenMock,
  }));
  const mockExpressRouter = { use: vi.fn(), post: vi.fn() };
  const mockExpressJson = vi.fn(() => vi.fn());
  return { mockExpressApp, mockExpressRouter, mockExpressJson, listenMock };
});

vi.mock("express", () => ({
  default: Object.assign(mockExpressApp, {
    Router: vi.fn(() => mockExpressRouter),
    json: mockExpressJson,
  }),
  Router: vi.fn(() => mockExpressRouter),
  json: mockExpressJson,
}));

describe("OpenAIServer config & warnings", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let capturedWarnings: string[];

  beforeEach(() => {
    capturedWarnings = [];
    listenMock.mockClear();
    warnSpy = vi.spyOn(logger, "warn").mockImplementation((msg: unknown) => {
      capturedWarnings.push(typeof msg === "string" ? msg : String(msg));
      return logger;
    });
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it("defaults host to loopback and leaves apiKey unset", () => {
    const config = new OpenAIServerConfig();
    expect(config.host).toBe("127.0.0.1");
    expect(config.apiKey).toBeUndefined();
  });

  it("warns when binding to a non-loopback host with no apiKey", async () => {
    const server = new OpenAIServer({ host: "0.0.0.0", apiKey: undefined });
    server.register(createDummyAgent(), { name: "dummy-model" });
    await server.serve();
    expect(hasInsecureWarning(capturedWarnings)).toBe(true);
  });

  it("does not warn on loopback host with no apiKey", async () => {
    const server = new OpenAIServer({ host: "127.0.0.1", apiKey: undefined });
    server.register(createDummyAgent(), { name: "dummy-model" });
    await server.serve();
    expect(hasInsecureWarning(capturedWarnings)).toBe(false);
  });

  it("does not warn on non-loopback host when apiKey is set", async () => {
    const server = new OpenAIServer({ host: "0.0.0.0", apiKey: "secret" });
    server.register(createDummyAgent(), { name: "dummy-model" });
    await server.serve();
    expect(hasInsecureWarning(capturedWarnings)).toBe(false);
  });
});
