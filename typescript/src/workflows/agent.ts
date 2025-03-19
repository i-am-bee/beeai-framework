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

import { Workflow, WorkflowRunOptions } from "@/workflows/workflow.js";
import { Message, UserMessage } from "@/backend/message.js";
import { AnyTool } from "@/tools/base.js";
import { BaseMemory, ReadOnlyMemory } from "@/memory/base.js";
import { z } from "zod";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { BaseAgent } from "@/agents/base.js";
import { isFunction, randomString } from "remeda";
import { ChatModel } from "@/backend/chat.js";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import {
  ToolCallingAgentExecutionConfig,
  ToolCallingAgentRunInput,
  ToolCallingAgentRunOptions,
  ToolCallingAgentRunOutput,
} from "@/agents/toolCalling/types.js";

type AgentInstance = BaseAgent<
  ToolCallingAgentRunInput,
  ToolCallingAgentRunOutput,
  ToolCallingAgentRunOptions
>;
type AgentFactory = (memory: ReadOnlyMemory) => AgentInstance | Promise<AgentInstance>;
interface AgentFactoryInput {
  name?: string;
  role?: string;
  llm: ChatModel;
  instructions?: string;
  tools?: AnyTool[];
  execution?: ToolCallingAgentExecutionConfig;
}

export class AgentWorkflow {
  protected readonly workflow;

  readonly schema = z.object({
    inputs: z.array(
      z.object({
        prompt: z.string().optional(),
        context: z.string().optional(),
        expectedOutput: z.union([z.string(), z.instanceof(z.ZodSchema)]).optional(),
      }),
    ),

    finalAnswer: z.string().optional(),
    newMessages: z.array(z.instanceof(Message)).default([]),
  });

  constructor(name = "AgentWorkflow") {
    this.workflow = new Workflow({
      name,
      schema: this.schema,
      outputSchema: this.schema.required(),
    });
  }

  run(inputs: (ToolCallingAgentRunInput | Message)[], options: WorkflowRunOptions<string> = {}) {
    return this.workflow.run(
      {
        inputs: inputs.map((input) => (input instanceof Message ? { prompt: input.text } : input)),
      },
      options,
    );
  }

  addAgent(agent: AgentFactory | AgentFactoryInput): this;
  addAgent(agent: AgentInstance): Promise<this>;
  addAgent(agent: AgentInstance | AgentFactory | AgentFactoryInput): this | Promise<this> {
    if (agent instanceof BaseAgent) {
      return agent.clone().then((clone) => {
        const factory: AgentFactory = (memory) => {
          clone.memory = memory;
          return clone;
        };
        return this._add(clone.meta.name, factory);
      });
    }

    const name = agent.name || `Agent${randomString(4)}`;
    return this._add(name, isFunction(agent) ? agent : this._createFactory(agent));
  }

  delAgent(name: string) {
    return this.workflow.delStep(name);
  }

  protected _createFactory(input: AgentFactoryInput): AgentFactory {
    return (memory: BaseMemory) =>
      new ToolCallingAgent({
        llm: input.llm,
        tools: input.tools ?? [],
        memory,
        meta: {
          name: input.name || `Agent${randomString(4)}`,
          description: input.instructions ?? "",
        },
        execution: input.execution,
        ...(input.instructions && {
          templates: {
            system: (template) =>
              template.fork((config) => {
                config.defaults.instructions = input.instructions || config.defaults.instructions;
                config.defaults.role = input.role || config.defaults.role;
              }),
          },
        }),
      });
  }

  protected _add(name: string, factory: AgentFactory) {
    this.workflow.addStep(name, async (state, ctx) => {
      const memory = new UnconstrainedMemory();
      await memory.addMany(state.newMessages);

      const runInput = state.inputs.shift() ?? { prompt: undefined };
      const agent = await factory(memory.asReadOnly());
      const { result } = await agent.run(runInput, { signal: ctx.signal });
      state.finalAnswer = result.text;
      if (runInput.prompt) {
        state.newMessages.push(new UserMessage(runInput.prompt));
      }
      state.newMessages.push(result);
    });
    return this;
  }
}
