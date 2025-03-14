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

import { ToolMessage } from "@/backend/message.js";
import type { AnyTool } from "@/tools/base.js";
import { DefaultRunner } from "@/agents/react/runners/default/runner.js";
import type { ReActAgentParserInput, ReActAgentRunOptions } from "@/agents/react/types.js";
import { ReActAgent, ReActAgentInput } from "@/agents/react/agent.js";
import type { GetRunContext } from "@/context.js";
import {
  GraniteReActAgentAssistantPrompt,
  GraniteReActAgentSchemaErrorPrompt,
  GraniteReActAgentSystemPrompt,
  GraniteReActAgentToolErrorPrompt,
  GraniteReActAgentToolInputErrorPrompt,
  GraniteReActAgentToolNotFoundPrompt,
  GraniteReActAgentUserPrompt,
} from "@/agents/react/runners/granite/prompts.js";
import {
  ReActAgentToolNoResultsPrompt,
  ReActAgentUserEmptyPrompt,
} from "@/agents/react/prompts.js";
import { Cache } from "@/cache/decoratorCache.js";

export class GraniteRunner extends DefaultRunner {
  protected useNativeToolCalling = true;

  @Cache({ enumerable: false })
  public get defaultTemplates() {
    return {
      system: GraniteReActAgentSystemPrompt,
      assistant: GraniteReActAgentAssistantPrompt,
      user: GraniteReActAgentUserPrompt,
      schemaError: GraniteReActAgentSchemaErrorPrompt,
      toolNotFoundError: GraniteReActAgentToolNotFoundPrompt,
      toolError: GraniteReActAgentToolErrorPrompt,
      toolInputError: GraniteReActAgentToolInputErrorPrompt,
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
      async ({ update, meta, memory, data }) => {
        if (update.key === "tool_output") {
          await memory.add(
            new ToolMessage(
              {
                type: "tool-result",
                result: update.value!,
                toolName: data.tool_name!,
                isError: !meta.success,
                toolCallId: "DUMMY_ID",
              },
              { success: meta.success },
            ),
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
      parser: parser.fork<ReActAgentParserInput>((nodes, options) => ({
        options,
        nodes: {
          ...nodes,
          thought: { ...nodes.thought, prefix: "Thought:" },
          tool_name: { ...nodes.tool_name, prefix: "Tool Name:" },
          tool_input: { ...nodes.tool_input, prefix: "Tool Input:", isEnd: true, next: [] },
          final_answer: { ...nodes.final_answer, prefix: "Final Answer:" },
        },
      })),
    };
  }
}
