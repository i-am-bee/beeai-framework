/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
