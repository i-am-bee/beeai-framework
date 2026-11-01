/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { z } from "zod";
import { ChatCompletionAPI } from "@/adapters/openai/serve/api.js";
import { ResponsesAPI } from "@/adapters/openai/serve/responses_api.js";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { HandoffTool } from "@/tools/handoff.js";
import { StringToolOutput, Tool, ToolEvents } from "@/tools/base.js";
import type { AnyAgent } from "@/agents/types.js";

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

/** A lead agent that hands off to an expert, which answers before the lead does. */
function createLeadAgent(): AnyAgent {
  const expert = new ToolCallingAgent({
    llm: new ScriptedModel([
      { toolName: "lookup", input: {} },
      { toolName: "final_answer", input: { response: "expert done" } },
    ]),
    tools: [new LookupTool()],
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
  // The handler clones the agent per request; reuse the scripted instance instead.
  vi.spyOn(lead, "clone").mockResolvedValue(lead);
  return lead as unknown as AnyAgent;
}

async function stream(api: ChatCompletionAPI | ResponsesAPI, body: object): Promise<string> {
  const req = Object.assign(new EventEmitter(), { body, headers: {} });
  const chunks: string[] = [];
  const res = {
    writableEnded: false,
    destroyed: false,
    setHeader: () => res,
    status: () => res,
    json: () => res,
    write: (chunk: string) => chunks.push(chunk),
    end: () => {
      res.writableEnded = true;
    },
  };
  await (api as any).handler(req, res);
  return chunks.join("");
}

describe("OpenAI-compatible server streaming", () => {
  it("streams only the lead agent's answer after a handoff (chat completions)", async () => {
    const api = new ChatCompletionAPI(async () => createLeadAgent());
    const output = await stream(api, {
      model: "lead",
      stream: true,
      messages: [{ role: "user", content: "go" }],
    });
    const deltas = output
      .split("\n\n")
      .filter((line) => line.startsWith("data: {"))
      .map((line) => JSON.parse(line.slice("data: ".length)).choices[0].delta.content)
      .filter(Boolean);

    expect(deltas).toEqual(["lead done"]);
  });

  it("streams only the lead agent's answer after a handoff (responses)", async () => {
    const api = new ResponsesAPI(async () => createLeadAgent());
    const output = await stream(api, { model: "lead", stream: true, input: "go" });
    const deltas = output
      .split("\n\n")
      .filter((event) => event.startsWith("event: response.output_text.delta\n"))
      .map((event) => JSON.parse(event.split("\ndata: ")[1]).delta);

    expect(deltas).toEqual(["lead done"]);
  });
});
