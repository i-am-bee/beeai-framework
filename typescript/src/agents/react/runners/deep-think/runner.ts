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

import type { AnyTool } from "@/tools/base.js";
import { DefaultRunner } from "@/agents/react/runners/default/runner.js";
import {
  DeepThinkReActAgentAssistantPrompt,
  DeepThinkReActAgentSchemaErrorPrompt,
  DeepThinkReActAgentSystemPrompt,
  DeepThinkReActAgentToolErrorPrompt,
  DeepThinkReActAgentToolInputErrorPrompt,
  DeepThinkReActAgentToolNotFoundPrompt,
  DeepThinkReActAgentUserPrompt,
} from "@/agents/react/runners/deep-think/prompts.js";
import {
  ReActAgentToolNoResultsPrompt,
  ReActAgentUserEmptyPrompt,
} from "@/agents/react/prompts.js";
import { Cache } from "@/cache/decoratorCache.js";
import { ZodParserField } from "@/parsers/field.js";
import { z } from "zod";
import { ReActAgentInput, ReActAgent } from "@/agents/react/agent.js";
import { ReActAgentRunOptions } from "@/agents/react/types.js";
import { GetRunContext } from "@/context.js";
import { UserMessage } from "@/backend/message.js";

export class DeepThinkRunner extends DefaultRunner {
  @Cache({ enumerable: false })
  public get defaultTemplates() {
    return {
      system: DeepThinkReActAgentSystemPrompt,
      assistant: DeepThinkReActAgentAssistantPrompt,
      user: DeepThinkReActAgentUserPrompt,
      schemaError: DeepThinkReActAgentSchemaErrorPrompt,
      toolNotFoundError: DeepThinkReActAgentToolNotFoundPrompt,
      toolError: DeepThinkReActAgentToolErrorPrompt,
      toolInputError: DeepThinkReActAgentToolInputErrorPrompt,
      // Note: These are from ReAct
      userEmpty: ReActAgentUserEmptyPrompt,
      toolNoResultError: ReActAgentToolNoResultsPrompt,
    };
  }

  static {
    this.register();
  }

  constructor(
    input: ReActAgentInput,
    options: ReActAgentRunOptions,
    run: GetRunContext<ReActAgent>,
  ) {
    super(input, options, run);

    run.emitter.on(
      "update",
      async ({ update, meta, memory }) => {
        if (update.key === "tool_output") {
          await memory.add(
            new UserMessage(update.value, { success: meta.success, createdAt: new Date() }),
          );
        }
      },
      {
        isBlocking: true,
      },
    );
  }

  protected createParser(tools: AnyTool[]) {
    const { parser } = super.createParser(tools);

    return {
      parser: parser.fork((nodes, options) => ({
        options: {
          ...options,
          // @ts-expect-error
          silentNodes: [...(options?.silentNodes ?? []), "dummy_thought_end"],
        },
        nodes: {
          ...nodes,
          thought: {
            ...nodes.thought,
            prefix: "<think>",
            // @ts-expect-error
            next: ["dummy_thought_end"] as const,
            isStart: true,
            field: new ZodParserField(z.string().min(1)),
          },
          dummy_thought_end: {
            prefix: "</think>",
            isDummy: true,
            next: ["tool_name", "final_answer"],
            field: new ZodParserField(z.string().transform((_) => "")),
          },
          tool_name: { ...nodes.tool_name, prefix: "Tool Name:" },
          tool_input: {
            ...nodes.tool_input,
            prefix: "Tool Input:",
            isEnd: true,
            next: [],
          },
          tool_output: { ...nodes.tool_name, prefix: "Tool Output:" },
          final_answer: { ...nodes.final_answer, prefix: "Response:" },
        },
      })),
    } as const;
  }
}
