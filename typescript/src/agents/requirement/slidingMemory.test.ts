/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { RequirementAgent } from "./agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, ToolMessage } from "@/backend/message.js";
import { Emitter } from "@/emitter/emitter.js";
import { SlidingMemory } from "@/memory/slidingMemory.js";

class ScriptedModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "sliding"], creator: this });
  requests: ChatModelInput[] = [];

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  async _create(input: ChatModelInput) {
    this.requests.push({ ...input, messages: input.messages.slice() });
    const n = this.requests.length;
    return new ChatModelOutput([
      new AssistantMessage({
        type: "tool-call",
        toolName: "final_answer",
        toolCallId: `answer-${n}`,
        input: { response: `answer ${n}` },
      }),
    ]);
  }

  async *_createStream(input: ChatModelInput) {
    yield await this._create(input);
  }
}

describe.each([true, false])("RequirementAgent sliding history (save: %s)", (save) => {
  it.each([4, 7, 10])("keeps tool exchanges paired with a window of %s", async (size) => {
    const model = new ScriptedModel();
    const memory = new SlidingMemory({ size });
    const agent = new RequirementAgent({
      llm: model,
      memory,
      saveIntermediateSteps: save,
    });
    try {
      for (let n = 1; n <= 6; n++) {
        expect((await agent.run({ prompt: `question ${n}` })).result.text).toBe(`answer ${n}`);
        expect(memory.messages.length).toBeLessThanOrEqual(size);
        const calls = new Set<string>();
        for (const msg of model.requests.at(-1)!.messages) {
          if (msg instanceof AssistantMessage) {
            msg.getToolCalls().forEach((entry) => calls.add(entry.toolCallId));
          } else if (msg instanceof ToolMessage) {
            for (const result of msg.getToolResults()) {
              expect(calls.has(result.toolCallId)).toBe(true);
            }
          }
        }
      }
    } finally {
      agent.destroy();
      model.destroy();
    }
  });
});
