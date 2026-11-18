/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { RequirementAgent } from "./agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, ToolMessage } from "@/backend/message.js";
import { Emitter } from "@/emitter/emitter.js";
import { DynamicTool, StringToolOutput } from "@/tools/base.js";

class ScriptedModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "history"], creator: this });
  requests: ChatModelInput[] = [];

  constructor(private readonly responses: AssistantMessage[][]) {
    super();
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  async _create(input: ChatModelInput) {
    this.requests.push({ ...input, messages: input.messages.slice() });
    const response = this.responses[this.requests.length - 1];
    if (!response) {
      throw new Error("Unexpected model request");
    }
    return new ChatModelOutput(response.slice());
  }

  async *_createStream(input: ChatModelInput) {
    yield await this._create(input);
  }
}

const call = (toolName: string, toolCallId: string, input: object) => ({
  type: "tool-call" as const,
  toolName,
  toolCallId,
  input,
});

describe.each([false, true])("RequirementAgent final exchange (stream: %s)", (stream) => {
  it.each(["single", "parallel", "reversed", "three", "separate messages"])(
    "retains the complete %s exchange without earlier intermediate steps",
    async (mode) => {
      const business = call("business", "business", { phase: "terminal" });
      const final = call("final_answer", "answer", { response: "first answer" });
      const calls =
        mode === "single"
          ? [final]
          : mode === "reversed"
            ? [final, business]
            : mode === "three"
              ? [business, call("business", "other", { phase: "other" }), final]
              : [business, final];
      const terminal =
        mode === "separate messages"
          ? calls.map(
              (entry, index) =>
                new AssistantMessage(entry, { source: "terminal" }, `terminal-${index}`),
            )
          : [
              new AssistantMessage(
                [{ type: "text", text: "Finished." }, ...calls],
                { source: "terminal" },
                "terminal-message",
              ),
            ];
      const originalContent = terminal.map((msg) => msg.content.slice());
      const model = new ScriptedModel([
        [new AssistantMessage(call("business", "early", { phase: "intermediate" }))],
        terminal,
        [new AssistantMessage(call("final_answer", "next-answer", { response: "second answer" }))],
      ]);
      model.parameters.stream = stream;
      const executions: string[] = [];
      const tool = new DynamicTool({
        name: "business",
        description: "Read a local fixture.",
        inputSchema: {
          type: "object",
          properties: { phase: { type: "string" } },
          required: ["phase"],
        },
        handler: async (input: { phase: string }) => {
          executions.push(input.phase);
          return new StringToolOutput("business result");
        },
      });
      const agent = new RequirementAgent({
        llm: model,
        tools: [tool],
        saveIntermediateSteps: false,
      });
      try {
        const first = await agent.run({ prompt: "first question" });
        expect(first.result.text).toBe("first answer");
        expect(first.state.steps.every((step) => !step.error)).toBe(true);
        const savedCalls = agent.memory.messages
          .filter((msg): msg is AssistantMessage => msg instanceof AssistantMessage)
          .flatMap((msg) => msg.getToolCalls().map((entry) => entry.toolCallId));
        const savedResults = agent.memory.messages
          .filter((msg): msg is ToolMessage => msg instanceof ToolMessage)
          .flatMap((msg) => msg.getToolResults().map((entry) => entry.toolCallId));
        const expected = calls.map((entry) => entry.toolCallId);
        expect(savedCalls).toEqual(expected);
        expect(savedResults).toEqual(expected);
        expect(agent.memory.messages[0].role).toBe("user");
        expect(agent.memory.messages).toHaveLength(1 + terminal.length + calls.length);
        expect(agent.memory.messages.slice(1, 1 + terminal.length)).toEqual(
          first.memory.messages.slice(-(terminal.length + calls.length), -calls.length),
        );
        expect(terminal.map((msg) => msg.content)).toEqual(originalContent);
        expect(executions).toEqual([
          "intermediate",
          ...(mode === "single" ? [] : ["terminal"]),
          ...(mode === "three" ? ["other"] : []),
        ]);

        const second = await agent.run({ prompt: "second question" });
        expect(second.result.text).toBe("second answer");
        expect(model.requests).toHaveLength(3);
        const history = model.requests[2].messages;
        expect(
          history
            .filter((msg): msg is AssistantMessage => msg instanceof AssistantMessage)
            .flatMap((msg) => msg.getToolCalls().map((entry) => entry.toolCallId)),
        ).toEqual(expected);
        expect(
          history
            .filter((msg): msg is ToolMessage => msg instanceof ToolMessage)
            .flatMap((msg) => msg.getToolResults().map((entry) => entry.toolCallId)),
        ).toEqual(expected);
      } finally {
        agent.destroy();
        model.destroy();
        tool.emitter.destroy();
      }
    },
  );
});
