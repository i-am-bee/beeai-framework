/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { RequirementAgent } from "./agent.js";
import { ReActAgent } from "@/agents/react/agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, Message, ToolMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { DynamicTool, JSONToolOutput } from "@/tools/base.js";
import { z } from "zod";

class RecordingModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "scripted"], creator: this });
  requests: ChatModelInput[] = [];

  constructor(private readonly react: boolean) {
    super();
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  async _create(input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
    this.requests.push({ ...input, messages: input.messages.slice() });
    const first = this.requests.length === 1;
    return new ChatModelOutput([
      new AssistantMessage(
        this.react
          ? first
            ? "Thought: Read the value.\nFunction Name: read_value\nFunction Input: {}"
            : "Thought: Finished.\nFinal Answer: done"
          : [
              {
                type: "tool-call",
                toolName: first ? "read_value" : "final_answer",
                toolCallId: first ? "read" : "answer",
                input: first ? {} : { response: "done" },
              },
            ],
      ),
    ]);
  }

  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}

function requirementResult(messages: readonly Message[]) {
  return messages
    .filter((message): message is ToolMessage => message instanceof ToolMessage)
    .flatMap((message) => message.getToolResults())
    .find((result) => result.toolName === "read_value")?.output;
}

describe.each([false, true])("JSON tool results (ReAct: %s)", (react) => {
  test.each([
    ...[0, -0, false, 1, true, -2, 0.5, "value", " \t\n", { value: false }, [0]].map((value) => ({
      value,
      text: JSON.stringify(value),
    })),
    ...[
      null,
      undefined,
      "",
      [],
      {},
      NaN,
      Infinity,
      -Infinity,
      0n,
      1n,
      new Boolean(false),
      new Number(0),
    ].map((value) => ({ value, text: null })),
  ])(
    "passes the expected result for $value to the next model request and history",
    async ({ value, text }) => {
      const model = new RecordingModel(react);
      const tool = new DynamicTool({
        name: "read_value",
        description: "Read a JSON value.",
        inputSchema: z.object({}),
        handler: async () => new JSONToolOutput(value),
      });
      const agent = react
        ? new ReActAgent({
            llm: model,
            tools: [tool],
            memory: new UnconstrainedMemory(),
            stream: false,
          })
        : new RequirementAgent({
            llm: model,
            tools: [tool],
            saveIntermediateSteps: true,
            finalAnswerAsTool: false,
          });
      try {
        const output = await agent.run({ prompt: "Read the value." });
        expect(output.result.text).toBe("done");
        expect(model.requests).toHaveLength(2);
        const expectedText =
          text ??
          (react
            ? "No results were found!"
            : "No results were found! Try to reformulate your query or use a different tool.");
        if (react) {
          expect(
            model.requests[1].messages.some((msg) =>
              msg.text.includes(`Function Output: ${expectedText}`),
            ),
          ).toBe(true);
          expect("iterations" in output && output.iterations[0].state.tool_output).toBe(
            expectedText,
          );
          expect(
            output.memory.messages.some((msg) =>
              msg.text.includes(`Function Output: ${expectedText}`),
            ),
          ).toBe(true);
        } else {
          const expected = { type: "text", value: expectedText };
          expect(requirementResult(model.requests[1].messages)).toEqual(expected);
          expect(requirementResult(agent.memory.messages)).toEqual(expected);
        }
      } finally {
        agent.destroy();
        model.destroy();
        tool.emitter.destroy();
      }
    },
  );
});
