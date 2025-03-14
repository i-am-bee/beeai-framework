/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { BaseAgent } from "@/agents/base.js";
import { AnyTool } from "@/tools/base.js";
import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { AgentMeta } from "@/agents/types.js";
import { Emitter } from "@/emitter/emitter.js";
import {
  ReActAgentExecutionConfig,
  ReActAgentTemplates,
  ReActAgentCallbacks,
  ReActAgentRunInput,
  ReActAgentRunOptions,
  ReActAgentRunOutput,
} from "@/agents/react/types.js";
import { GetRunContext } from "@/context.js";
import { assign } from "@/internals/helpers/object.js";
import * as R from "remeda";
import { BaseRunner } from "@/agents/react/runners/base.js";
import { GraniteRunner } from "@/agents/react/runners/granite/runner.js";
import { DeepThinkRunner } from "@/agents/react/runners/deep-think/runner.js";
import { ValueError } from "@/errors.js";
import { DefaultRunner } from "@/agents/react/runners/default/runner.js";
import { ChatModel } from "@/backend/chat.js";

export type ReActAgentTemplateFactory<K extends keyof ReActAgentTemplates> = (
  template: ReActAgentTemplates[K],
) => ReActAgentTemplates[K];

export interface ReActAgentInput {
  llm: ChatModel;
  tools: AnyTool[];
  memory: BaseMemory;
  meta?: Omit<AgentMeta, "tools">;
  templates?: Partial<{
    [K in keyof ReActAgentTemplates]: ReActAgentTemplates[K] | ReActAgentTemplateFactory<K>;
  }>;
  execution?: ReActAgentExecutionConfig;
  stream?: boolean;
}

export class ReActAgent extends BaseAgent<
  ReActAgentRunInput,
  ReActAgentRunOutput,
  ReActAgentRunOptions
> {
  public readonly emitter = Emitter.root.child<ReActAgentCallbacks>({
    namespace: ["agent", "react"],
    creator: this,
  });

  protected runner: new (...args: ConstructorParameters<typeof BaseRunner>) => BaseRunner;

  constructor(protected readonly input: ReActAgentInput) {
    super();

    const duplicate = input.tools.find((a, i, arr) =>
      arr.find((b, j) => i !== j && a.name.toUpperCase() === b.name.toUpperCase()),
    );
    if (duplicate) {
      throw new ValueError(
        `Agent's tools must all have different names. Conflicting tool: ${duplicate.name}.`,
      );
    }

    const modelId = this.input.llm.modelId.toLowerCase();
    this.runner =
      [
        { tag: "granite", runner: GraniteRunner },
        { tag: "deepseek-r1", runner: DeepThinkRunner },
      ].find(({ tag }) => modelId.includes(tag))?.runner ?? DefaultRunner;
  }

  static {
    this.register();
  }

  set memory(memory: BaseMemory) {
    this.input.memory = memory;
  }

  get memory() {
    return this.input.memory;
  }

  get meta(): AgentMeta {
    const tools = this.input.tools.slice();

    if (this.input.meta) {
      return { ...this.input.meta, tools };
    }

    return {
      name: "ReAct",
      tools,
      description:
        "The Bee framework demonstrates its ability to auto-correct and adapt in real-time, improving the overall reliability and resilience of the system.",
      ...(tools.length > 0 && {
        extraDescription: [
          `Tools that I can use to accomplish given task.`,
          ...tools.map((tool) => `Tool '${tool.name}': ${tool.description}.`),
        ].join("\n"),
      }),
    };
  }

  protected async _run(
    input: ReActAgentRunInput,
    options: ReActAgentRunOptions = {},
    run: GetRunContext<typeof this>,
  ): Promise<ReActAgentRunOutput> {
    const runner = new this.runner(
      this.input,
      {
        ...options,
        execution: this.input.execution ??
          options?.execution ?? {
            maxRetriesPerStep: 3,
            totalMaxRetries: 20,
            maxIterations: 10,
          },
      },
      run,
    );
    await runner.init(input);

    let finalMessage: Message | undefined;
    while (!finalMessage) {
      const { state, meta, emitter, signal } = await runner.createIteration();

      if (state.tool_name && state.tool_input) {
        const { output, success } = await runner.tool({
          state,
          emitter,
          meta,
          signal,
        });
        await runner.memory.add(
          new AssistantMessage(
            runner.templates.assistant.render({
              thought: [state.thought].filter(R.isTruthy),
              toolName: [state.tool_name].filter(R.isTruthy),
              toolInput: [state.tool_input].filter(R.isTruthy).map((call) => JSON.stringify(call)),
              toolOutput: [output],
              finalAnswer: [state.final_answer].filter(R.isTruthy),
            }),
            { success },
          ),
        );
        assign(state, { tool_output: output });

        for (const key of ["partialUpdate", "update"] as const) {
          await emitter.emit(key, {
            data: state,
            update: { key: "tool_output", value: output, parsedValue: output },
            meta: { success, ...meta },
            memory: runner.memory,
          });
        }
      }
      if (state.final_answer) {
        finalMessage = new AssistantMessage(state.final_answer, {
          createdAt: new Date(),
        });
        await runner.memory.add(finalMessage);
        await emitter.emit("success", {
          data: finalMessage,
          iterations: runner.iterations,
          memory: runner.memory,
          meta,
        });
      }
    }

    if (input.prompt !== null) {
      await this.input.memory.add(
        new UserMessage(input.prompt, {
          createdAt: run.createdAt,
        }),
      );
    }

    await this.input.memory.add(finalMessage);
    return { result: finalMessage, iterations: runner.iterations, memory: runner.memory };
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: this.input,
      emitter: this.emitter,
      runner: this.runner,
    };
  }
}
