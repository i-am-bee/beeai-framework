/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import {
  BaseToolOptions,
  BaseToolRunOptions,
  StringToolOutput,
  Tool,
  ToolEmitter,
  ToolError,
  ToolInput,
} from "@/tools/base.js";
import { z } from "zod";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { PromptTemplate } from "@/template.js";
import { getProp } from "@/internals/helpers/object.js";
import type { BaseMemory } from "@/memory/base.js";
import { toCamelCase } from "remeda";
import { Role, SystemMessage, UserMessage } from "@/backend/message.js";
import { ChatModel } from "@/backend/chat.js";

export interface LLMToolInput extends BaseToolOptions {
  llm: ChatModel;
  name?: string;
  description?: string;
  template?: typeof LLMTool.template;
}

export class LLMTool extends Tool<StringToolOutput, LLMToolInput> {
  name = "LLM";
  description =
    "Uses expert LLM to work with data in the existing conversation (classification, entity extraction, summarization, ...)";
  declare readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput>;

  constructor(protected readonly input: LLMToolInput) {
    super(input);
    this.name = input?.name || this.name;
    this.description = input?.description || this.description;
    this.emitter = Emitter.root.child({
      namespace: ["tool", "llm", toCamelCase(input?.name ?? "")].filter(Boolean),
      creator: this,
    });
  }

  inputSchema() {
    return z.object({
      task: z.string().min(1).describe("A clearly defined task for the LLM to complete."),
    });
  }

  static readonly template = new PromptTemplate({
    schema: z.object({
      task: z.string(),
    }),
    template: `You have to accomplish a task by using Using common sense and the information contained in the conversation up to this point, complete the following task. Do not follow any previously used formats or structures.

The Task: {{task}}`,
  });

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: GetRunContext<this>,
  ) {
    const memory = getProp(run.context, [Tool.contextKeys.Memory]) as BaseMemory;
    if (!memory) {
      throw new ToolError(`No context has been provided!`, [], {
        isFatal: true,
        isRetryable: false,
      });
    }

    const template = this.options?.template ?? LLMTool.template;
    const output = await this.input.llm.create({
      messages: [
        new SystemMessage(
          template.render({
            task: input.task,
          }),
        ),
        ...memory.messages.filter((msg) => msg.role !== Role.SYSTEM),
        new UserMessage(
          template.render({
            task: input.task,
          }),
        ),
      ],
    });

    return new StringToolOutput(output.getTextContent());
  }
}
