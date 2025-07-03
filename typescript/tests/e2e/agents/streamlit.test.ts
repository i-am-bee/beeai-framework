/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";
import { beforeEach, expect, vi } from "vitest";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { createCallbackRegister } from "@tests/e2e/utils.js";
import { createChatLLM } from "@tests/utils/llmFactory.js";
import { StreamlitAgent } from "@/agents/experimental/streamlit/agent.js";
import { BaseMemory } from "@/memory/base.js";
import { Message } from "@/backend/message.js";
import { ChatModelOutput } from "@/backend/chat.js";

describe("Streamlit Agent", () => {
  const createAgent = () => {
    return new StreamlitAgent({
      llm: createChatLLM(),
      memory: new UnconstrainedMemory(),
    });
  };

  beforeEach(() => {
    vi.useRealTimers();
  });

  it("Runs", async () => {
    const callbacks = createCallbackRegister();
    const agent = createAgent();

    let response: Awaited<ReturnType<typeof agent.run>>;
    try {
      response = await agent
        .run({
          prompt: `Generate a minimalistic "Hello World" app and provide some short explanation.`,
        })
        .observe((emitter) => {
          emitter.registerCallbacks({
            newToken: callbacks.create("newToken", {
              required: true,
              check: (value) => {
                expect(value).toBeTruthy();
                expect(value.delta).toBeTypeOf("string");
                expect(value.state).toBeTypeOf("object");
                expect(value.chunk).toBeInstanceOf(ChatModelOutput);
              },
            }),
          });
        });
    } catch (e) {
      expect(e).toBeInstanceOf(FrameworkError);
      throw e;
    } finally {
      agent.destroy();
    }

    expect(response.memory).toBeInstanceOf(BaseMemory);
    expect(response.message).toBeInstanceOf(Message);

    expect(response.result.raw).toBeTypeOf("string");
    expect(response.result.blocks.length).toBeGreaterThanOrEqual(2);

    const stats = { app: 0, text: 0 };
    response.result.blocks.forEach((block, i, arr) => {
      const prevBlock = arr[i - 1];
      if (prevBlock) {
        expect(prevBlock.end).toBe(block.start);
        expect(block.start).toBe(prevBlock.end);
      } else {
        expect(block.start).toBe(0);
      }
      expect(block.content).toBeTruthy();
      if (block.name === "app") {
        stats.app++;
      } else {
        stats.text++;
      }
    });
    expect(stats.app).toBeGreaterThan(0);
    expect(stats.text).toBeGreaterThan(0);

    callbacks.verify();
  });
});
