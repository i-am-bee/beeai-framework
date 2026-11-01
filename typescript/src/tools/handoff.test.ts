/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage } from "@/backend/message.js";
import { GetRunContext, RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { GlobalTrajectoryMiddleware } from "@/middleware/trajectory.js";
import { HandoffTool } from "@/tools/handoff.js";
import { StringToolOutput, Tool, ToolEvents } from "@/tools/base.js";
import { z } from "zod";

class LookupTool extends Tool {
  name = "lookup";
  description = "Look something up";
  emitter = Emitter.root.child<ToolEvents>({ namespace: ["tool", "lookup"], creator: this });
  inputSchema() {
    return z.object({});
  }
  async _run() {
    return new StringToolOutput("found");
  }
}

/** Replies with the next scripted tool call on each model call. */
class ScriptedModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "scripted"], creator: this });
  calls = 0;
  constructor(private readonly script: { toolName: string; input: object }[] = []) {
    super();
  }
  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
  async _create(_input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
    const step = this.script[this.calls++];
    return new ChatModelOutput([
      new AssistantMessage([
        {
          type: "tool-call",
          toolName: step.toolName,
          toolCallId: `call-${this.calls}`,
          input: step.input,
        },
      ]),
    ]);
  }
  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}

function createAgents() {
  const lookup = new LookupTool();
  const expert = new ToolCallingAgent({
    llm: new ScriptedModel([
      { toolName: "lookup", input: {} },
      { toolName: "final_answer", input: { response: "expert done" } },
    ]),
    tools: [lookup],
    memory: new UnconstrainedMemory(),
    meta: { name: "Expert", description: "Looks things up." },
  });
  const handoff = new HandoffTool(expert);
  const lead = new ToolCallingAgent({
    llm: new ScriptedModel([
      { toolName: handoff.name, input: { task: "find it" } },
      { toolName: "final_answer", input: { response: "lead done" } },
    ]),
    tools: [handoff],
    memory: new UnconstrainedMemory(),
  });

  return {
    lead,
    destroy: () => {
      lead.destroy();
      expert.destroy();
      handoff.emitter.destroy();
      lookup.emitter.destroy();
    },
  };
}

describe("HandoffTool nested run events", () => {
  it("Forwards a sub-agent's tool calls to a nested listener on the outer run", async () => {
    const { lead, destroy } = createAgents();
    const lookups: string[] = [];

    try {
      const result = await lead.run({ prompt: "go" }).observe((emitter) => {
        emitter.match(
          (event) =>
            event.name === "start" &&
            event.creator instanceof RunContext &&
            event.creator.instance instanceof LookupTool,
          (_, event) => {
            lookups.push(event.path);
          },
          { matchNested: true },
        );
      });

      expect(result.result.text).toBe("lead done");
      expect(lookups).toHaveLength(1);
    } finally {
      destroy();
    }
  });

  it("Logs each event of a two-level handoff once in the trajectory", async () => {
    const { lead, destroy } = createAgents();
    const lines: string[] = [];

    try {
      await lead
        .run({ prompt: "go" })
        .middleware(
          new GlobalTrajectoryMiddleware({ included: [Tool], target: (line) => lines.push(line) }),
        );

      expect(lines.some((line) => line.includes("LookupTool"))).toBe(true);
      expect(lines.length).toBe(new Set(lines).size);
    } finally {
      destroy();
    }
  });
});
