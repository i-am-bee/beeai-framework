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

import { AgentError, BaseAgent } from "@/agents/base.js";
import { AnyTool, ToolError, ToolOutput } from "@/tools/base.js";
import { BaseMemory } from "@/memory/base.js";
import { AgentMeta } from "@/agents/types.js";
import { Emitter } from "@/emitter/emitter.js";
import type {
  ToolCallingAgentExecutionConfig,
  ToolCallingAgentTemplates,
  ToolCallingAgentCallbacks,
  ToolCallingAgentRunInput,
  ToolCallingAgentRunOptions,
  ToolCallingAgentRunOutput,
  ToolCallingAgentRunState,
} from "@/agents/toolCalling/types.js";
import { GetRunContext } from "@/context.js";
import { ChatModel } from "@/backend/chat.js";
import { shallowCopy } from "@/serializer/utils.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { AssistantMessage, SystemMessage, ToolMessage, UserMessage } from "@/backend/message.js";
import { isEmpty, isString } from "remeda";
import { RetryCounter } from "@/internals/helpers/counter.js";
import { mapObj, omitUndefined } from "@/internals/helpers/object.js";
import { Cache } from "@/cache/decoratorCache.js";
import { PromptTemplate } from "@/template.js";
import {
  ToolCallingAgentSystemPrompt,
  ToolCallingAgentTaskPrompt,
} from "@/agents/toolCalling/prompts.js";
import { ZodSchema } from "zod";

export type ToolCallingAgentTemplateFactory<K extends keyof ToolCallingAgentTemplates> = (
  template: ToolCallingAgentTemplates[K],
) => ToolCallingAgentTemplates[K];

export interface ToolCallingAgentInput {
  llm: ChatModel;
  memory: BaseMemory;
  tools: AnyTool[];
  meta?: Omit<AgentMeta, "tools">;
  templates?: Partial<{
    [K in keyof ToolCallingAgentTemplates]:
      | ToolCallingAgentTemplates[K]
      | ToolCallingAgentTemplateFactory<K>;
  }>;
  execution?: ToolCallingAgentExecutionConfig;
}

export class ToolCallingAgent extends BaseAgent<
  ToolCallingAgentRunInput,
  ToolCallingAgentRunOutput,
  ToolCallingAgentRunOptions
> {
  public readonly emitter = Emitter.root.child<ToolCallingAgentCallbacks>({
    namespace: ["agent", "toolCalling"],
    creator: this,
  });

  constructor(public readonly input: ToolCallingAgentInput) {
    super();
  }

  static {
    this.register();
  }

  protected async _run(
    input: ToolCallingAgentRunInput,
    options: ToolCallingAgentRunOptions = {},
    run: GetRunContext<typeof this>,
  ): Promise<ToolCallingAgentRunOutput> {
    const tempMessageKey = "tempMessage" as const;
    const execution = {
      maxRetriesPerStep: 3,
      totalMaxRetries: 20,
      maxIterations: 10,
      ...omitUndefined(this.input.execution ?? {}),
      ...omitUndefined(options.execution ?? {}),
    };

    const state: ToolCallingAgentRunState = {
      memory: new UnconstrainedMemory(),
      result: undefined,
      iteration: 0,
    };
    await state.memory.add(
      new SystemMessage(
        this.templates.system.render({
          role: undefined,
          instructions: undefined,
        }),
      ),
    );

    if (input.prompt) {
      const userMessage = new UserMessage(
        this.templates.task.render({
          prompt: input.prompt,
          context: input.context,
          expectedOutput: isString(input.expectedOutput) ? input.expectedOutput : undefined,
        }),
      );
      await state.memory.add(userMessage);
    }

    const globalRetriesCounter = new RetryCounter(execution.totalMaxRetries || 1, AgentError);

    while (!state.result) {
      state.iteration++;
      if (state.iteration > (execution.totalMaxRetries ?? Infinity)) {
        throw new AgentError(
          `Agent was not able to resolve the task in ${state.iteration} iterations.`,
        );
      }

      await run.emitter.emit("start", { state });
      const response = await this.input.llm.create({
        messages: state.memory.messages.slice(),
        tools: this.input.tools,
        stream: false,
      });
      await state.memory.addMany(response.messages);

      const toolCallMessages = response.getToolCalls();
      for (const toolCall of toolCallMessages) {
        try {
          const tool = this.input.tools.find((tool) => tool.name === toolCall.toolName);
          if (!tool) {
            throw new AgentError(`Tool ${toolCall.toolName} does not exist!`);
          }

          const toolInput: any = toolCall.args;
          const toolResponse: ToolOutput = await tool.run(toolInput).context({
            state,
            toolCallMsg: toolCall,
          });
          await state.memory.add(
            new ToolMessage({
              type: "tool-result",
              toolCallId: toolCall.toolCallId,
              toolName: toolCall.toolName,
              result: toolResponse.getTextContent(),
              isError: false,
            }),
          );
        } catch (e) {
          if (e instanceof ToolError) {
            globalRetriesCounter.use(e);
            await state.memory.add(
              new ToolMessage({
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                result: e.explain(),
                isError: true,
              }),
            );
          } else {
            throw e;
          }
        }
      }

      // handle empty messages for some models
      const textMessages = response.getTextMessages();
      if (isEmpty(toolCallMessages) && isEmpty(textMessages)) {
        await state.memory.add(new AssistantMessage("\n", { [tempMessageKey]: true }));
      } else {
        await state.memory.deleteMany(
          state.memory.messages.filter((msg) => msg.meta[tempMessageKey]),
        );
      }

      if (!isEmpty(textMessages) && isEmpty(toolCallMessages)) {
        if (input.expectedOutput && input.expectedOutput instanceof ZodSchema) {
          const structured = await this.input.llm.createStructure({
            schema: input.expectedOutput,
            messages: state.memory.messages.slice(),
          });
          state.result = new AssistantMessage(JSON.stringify(structured, null, 4));
        } else {
          state.result = AssistantMessage.fromChunks(textMessages);
        }
      }

      await run.emitter.emit("success", { state });
    }

    await this.memory.addMany(state.memory.messages.slice(1));
    return { memory: state.memory, result: state.result };
  }

  get meta(): AgentMeta {
    const tools = this.input.tools.slice();

    if (this.input.meta) {
      return { ...this.input.meta, tools };
    }

    return {
      name: "ToolCalling",
      tools,
      description: "ToolCallingAgent that uses tools to accomplish the task.",
      ...(tools.length > 0 && {
        extraDescription: [
          `Tools that I can use to accomplish given task.`,
          ...tools.map((tool) => `Tool '${tool.name}': ${tool.description}.`),
        ].join("\n"),
      }),
    };
  }

  @Cache({ enumerable: false })
  protected get templates(): ToolCallingAgentTemplates {
    const overrides = this.input.templates ?? {};
    const defaultTemplates: ToolCallingAgentTemplates = {
      system: ToolCallingAgentSystemPrompt,
      task: ToolCallingAgentTaskPrompt,
    } as const;

    return mapObj(defaultTemplates)(
      (key, defaultTemplate: ToolCallingAgentTemplates[typeof key]) => {
        const override = overrides[key] ?? defaultTemplate;
        if (override instanceof PromptTemplate) {
          return override;
        }
        return override(defaultTemplate) ?? defaultTemplate;
      },
    );
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      emitter: this.emitter,
    };
  }

  set memory(memory: BaseMemory) {
    this.input.memory = memory;
  }

  get memory() {
    return this.input.memory;
  }
}
