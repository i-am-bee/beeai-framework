/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { ToolCallingAgent } from "./agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { Tool, ToolEvents, StringToolOutput } from "@/tools/base.js";
import { z } from "zod";

class BusinessTool extends Tool {
  name = "business";
  description = "Continue processing";
  emitter = Emitter.root.child<ToolEvents>({ namespace: ["tool", "business"], creator: this });
  inputSchema() {
    return z.object({});
  }
  async _run() {
    return new StringToolOutput("continue");
  }
}

class ScriptedModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "scriptedToolCalling"], creator: this });
  calls = 0;
  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
  async _create(_input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
    this.calls++;
    return new ChatModelOutput([
      new AssistantMessage([
        {
          type: "tool-call",
          toolName: this.calls < 3 ? "business" : "final_answer",
          toolCallId: `call-${this.calls}`,
          input: this.calls < 3 ? {} : { response: "done" },
        },
      ]),
    ]);
  }
  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}

async function withAgent(
  execution: { maxIterations?: number; totalMaxRetries?: number },
  run: (agent: ToolCallingAgent, model: ScriptedModel) => Promise<void>,
) {
  const model = new ScriptedModel();
  const tool = new BusinessTool();
  const agent = new ToolCallingAgent({
    llm: model,
    tools: [tool],
    memory: new UnconstrainedMemory(),
    execution,
  });
  try {
    await run(agent, model);
  } finally {
    agent.destroy();
    model.destroy();
    tool.emitter.destroy();
  }
}

describe("ToolCallingAgent iteration limits", () => {
  it("makes no model call when maxIterations is zero", async () => {
    await withAgent({ maxIterations: 0 }, async (agent, model) => {
      await expect(agent.run({ prompt: "hello" })).rejects.toThrow(/1 iterations/);
      expect(model.calls).toBe(0);
    });
  });

  it("stops after the configured number of model calls, not totalMaxRetries", async () => {
    await withAgent({ maxIterations: 2, totalMaxRetries: 5 }, async (agent, model) => {
      await expect(agent.run({ prompt: "hello" })).rejects.toThrow(/3 iterations/);
      expect(model.calls).toBe(2);
    });
  });

  it("does not use totalMaxRetries as an iteration limit", async () => {
    await withAgent({ totalMaxRetries: 1 }, async (agent, model) => {
      const result = await agent.run({ prompt: "hello" });
      expect(result.result.text).toBe("done");
      expect(model.calls).toBe(3);
    });
  });
});
