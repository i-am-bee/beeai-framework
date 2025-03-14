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

import { ReActAgent } from "@/agents/react/agent.js";
import { Workflow, WorkflowRunOptions } from "@/workflows/workflow.js";
import { AssistantMessage, Message } from "@/backend/message.js";
import { AnyTool } from "@/tools/base.js";
import { BaseMemory, ReadOnlyMemory } from "@/memory/base.js";
import { z } from "zod";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { BaseAgent } from "@/agents/base.js";
import {
  ReActAgentExecutionConfig,
  ReActAgentRunInput,
  ReActAgentRunOptions,
  ReActAgentRunOutput,
} from "@/agents/react/types.js";
import { isFunction, randomString } from "remeda";
import { ChatModel } from "@/backend/chat.js";

type AgentInstance = BaseAgent<ReActAgentRunInput, ReActAgentRunOutput, ReActAgentRunOptions>;
type AgentFactory = (memory: ReadOnlyMemory) => AgentInstance | Promise<AgentInstance>;
interface AgentFactoryInput {
  name: string;
  llm: ChatModel;
  instructions?: string;
  tools?: AnyTool[];
  execution?: ReActAgentExecutionConfig;
}

export class AgentWorkflow {
  protected readonly workflow;

  static readonly schema = z.object({
    messages: z.array(z.instanceof(Message)).min(1),

    finalAnswer: z.string().optional(),
    newMessages: z.array(z.instanceof(Message)).default([]),
  });

  constructor(name = "AgentWorkflow") {
    this.workflow = new Workflow({
      name,
      schema: AgentWorkflow.schema,
      outputSchema: AgentWorkflow.schema.required(),
    });
  }

  run(messages: Message[], options: WorkflowRunOptions<string> = {}) {
    return this.workflow.run(
      {
        messages,
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
      new ReActAgent({
        llm: input.llm,
        tools: input.tools ?? [],
        memory,
        meta: {
          name: input.name,
          description: input.instructions ?? "",
        },
        execution: input.execution,
        ...(input.instructions && {
          templates: {
            system: (template) =>
              template.fork((config) => {
                config.defaults.instructions = input.instructions || config.defaults.instructions;
              }),
          },
        }),
      });
  }

  protected _add(name: string, factory: AgentFactory) {
    this.workflow.addStep(name, async (state, ctx) => {
      const memory = new UnconstrainedMemory();
      await memory.addMany([...state.messages, ...state.newMessages]);

      const agent = await factory(memory.asReadOnly());
      const { result } = await agent.run({ prompt: null }, { signal: ctx.signal });
      state.finalAnswer = result.text;
      state.newMessages.push(
        new AssistantMessage(
          [`Assistant Name: ${name}`, `Assistant Response: ${result.text}`].join("\n"),
        ),
      );
    });
    return this;
  }
}
