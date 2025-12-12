/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AnyTool,
  BaseToolRunOptions,
  StringToolOutput,
  Tool,
  ToolError,
  ToolOutput,
} from "@/tools/base.js";
import { ToolCallPart } from "ai";
import { FrameworkError } from "@/errors.js";
import { Emitter } from "@/emitter/emitter.js";
import { z, ZodSchema } from "zod";
import { RequirementAgentRunState } from "@/agents/requirement/types.js";
import { AssistantMessage } from "@/backend/message.js";
import { RunContext } from "@/context.js";

// Tool invocation result
export interface ToolInvocationResult {
  msg: ToolCallPart;
  tool: AnyTool | null;
  input: unknown;
  output: ToolOutput;
  error: FrameworkError | null;
}

/**
 * Run a single tool with error handling
 */
export async function runTool(
  tools: AnyTool[],
  msg: ToolCallPart,
  context: Record<string, any>,
): Promise<ToolInvocationResult> {
  const result: ToolInvocationResult = {
    msg,
    tool: null,
    input: msg.input,
    output: new StringToolOutput(""),
    error: null,
  };

  try {
    result.tool = tools.find((tool) => tool.name === msg.toolName) || null;
    if (!result.tool) {
      throw new ToolError(`Tool '${msg.toolName}' does not exist!`);
    }

    result.output = await result.tool.run(result.input).context({
      ...context,
      toolCallMsg: msg,
    });
  } catch (e) {
    if (e instanceof ToolError) {
      result.error = FrameworkError.ensure(e);
    } else {
      throw e;
    }
  }

  return result;
}

// Final answer tool schema
const FinalAnswerToolSchema = z.object({
  response: z.string().describe("The final answer to the user"),
});

/**
 * Special tool for capturing final answers
 */
export class FinalAnswerTool extends Tool {
  public readonly name = "final_answer";
  public readonly description = "Sends the final answer to the user";
  public instructions?: string;
  public customSchema: boolean;
  public readonly emitter: Emitter<any>;

  protected expectedOutput: string | ZodSchema | null;
  protected state: RequirementAgentRunState;

  constructor(expectedOutput: string | ZodSchema | null, state: RequirementAgentRunState) {
    super();
    this.expectedOutput = expectedOutput;
    this.state = state;
    this.instructions = typeof expectedOutput === "string" ? expectedOutput : undefined;
    this.customSchema = expectedOutput instanceof ZodSchema;
    this.emitter = Emitter.root.child({
      namespace: ["tool", "final_answer"],
      creator: this,
    });
  }

  inputSchema(): ZodSchema {
    const expectedOutput = this.expectedOutput;

    if (!expectedOutput) {
      return FinalAnswerToolSchema;
    } else if (expectedOutput instanceof ZodSchema) {
      return expectedOutput;
    } else if (typeof expectedOutput === "string") {
      return z.object({
        response: z.string().describe(expectedOutput),
      });
    } else {
      return FinalAnswerToolSchema;
    }
  }

  async _run(
    input: any,
    _options: BaseToolRunOptions,
    _ctx: RunContext<typeof this>,
  ): Promise<StringToolOutput> {
    this.state.result = input;

    if (this.expectedOutput instanceof ZodSchema) {
      // For custom schemas, serialize the entire input
      this.state.answer = new AssistantMessage(JSON.stringify(input));
    } else {
      // For string schemas, use the response field
      this.state.answer = new AssistantMessage(input.response);
    }

    return new StringToolOutput("Message has been sent");
  }
}
