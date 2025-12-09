/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentError, BaseAgent } from "@/agents/base.js";
import { AnyTool, DynamicTool, StringToolOutput, ToolError, ToolOutput } from "@/tools/base.js";
import { BaseMemory } from "@/memory/base.js";
import { AgentMeta } from "@/agents/types.js";
import { Emitter } from "@/emitter/emitter.js";
import type {
  RequirementAgentExecutionConfig,
  RequirementAgentTemplates,
  RequirementAgentCallbacks,
  RequirementAgentRunInput,
  RequirementAgentRunOptions,
  RequirementAgentRunOutput,
  RequirementAgentRunState,
} from "@/agents/requirement/types.js";
import { GetRunContext } from "@/context.js";
import { ChatModel } from "@/backend/chat.js";
import { shallowCopy } from "@/serializer/utils.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { AssistantMessage, SystemMessage, ToolMessage, UserMessage } from "@/backend/message.js";
import { isEmpty, isString } from "remeda";
import { RetryCounter } from "@/internals/helpers/counter.js";
import { omitUndefined } from "@/internals/helpers/object.js";

import { z, ZodSchema } from "zod";
import { createTemplates } from "@/agents/requirement/utils.js";
import { RequirementAgentRunner } from "@/agents/requirement/runner.js";

export type RequirementAgentTemplateFactory<K extends keyof RequirementAgentTemplates> = (
  template: RequirementAgentTemplates[K],
) => RequirementAgentTemplates[K];

export interface RequirementAgentInput {
  llm: ChatModel;
  memory?: BaseMemory;
  tools?: AnyTool[];
  requirements?: RequirementAgentRequirement[];
  name?: string;
  description?: string;
  instructions?: string | string[];
  notes?: string | string[];
  toolCallChecker?: ToolCallCheckerConfig[];
  finalAnswerAsTool?: boolean;
  saveIntermediateSteps?: boolean;
  templates?: Partial<{
    [K in keyof RequirementAgentTemplates]:
      | RequirementAgentTemplates[K]
      | RequirementAgentTemplateFactory<K>;
  }>;
  execution?: RequirementAgentExecutionConfig;
  middlewares?: MiddlewareFn[];
}

export class RequirementAgent extends BaseAgent<
  RequirementAgentRunInput,
  RequirementAgentRunOutput,
  RequirementAgentRunOptions
> {
  /**
   * The RequirementAgent is a declarative AI agent implementation that provides predictable,
   * controlled execution behavior across different language models through rule-based constraints.
   * Language models vary significantly in their reasoning capabilities and tool-calling sophistication, but
   * RequirementAgent normalizes these differences by enforcing consistent execution patterns
   * regardless of the underlying model's strengths or weaknesses.
   * Rules can be configured as strict or flexible as necessary, adapting to task requirements while ensuring consistent
   * execution regardless of the underlying model's reasoning or tool-calling capabilities.
   */

  protected readonly input: Required<RequirementAgentInput>;
  protected runner: new (
    ...args: ConstructorParameters<typeof RequirementAgentRunner>
  ) => RequirementAgent;

  public readonly emitter = Emitter.root.child<RequirementAgentCallbacks>({
    namespace: ["agent", "requirement"],
    creator: this,
  });

  constructor(input: RequirementAgentInput) {
    super();
    this.input = {
      ...input,
      memory: input.memory ?? new UnconstrainedMemory(),
      tools: input.tools ?? [],
      saveIntermediateSteps: input.saveIntermediateSteps ?? true,
      requirements: input.requirements ?? [],
      finalAnswerAsTool: input.finalAnswerAsTool ?? true,
      execution: input.execution ?? {},
      templates: createTemplates(input.templates),
      name: "",
      description: "",
      instructions: input.instructions ?? [],
      notes: input.notes ?? [],
      toolCallChecker: input.toolCallChecker ?? {},
    };
    this.runner = RequirementAgentRunner;
  }

  static {
    this.register();
  }

  protected async _run(
    input: RequirementAgentRunInput,
    options: RequirementAgentRunOptions = {},
    run: GetRunContext<typeof this>,
  ): Promise<RequirementAgentRunOutput> {
    const tempMessageKey = "tempMessage" as const;
    const execution = {
      maxRetriesPerStep: 3,
      totalMaxRetries: 20,
      maxIterations: 10,
      ...omitUndefined(this.input.execution ?? {}),
      ...omitUndefined(options.execution ?? {}),
    };

    const state: RequirementAgentRunState = {
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
    await state.memory.addMany(this.memory.messages);

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

    const usePlainResponse = !input.expectedOutput || !(input.expectedOutput instanceof ZodSchema);
    const finalAnswerToolSchema = usePlainResponse
      ? z.object({
          response: z.string().describe(String(input.expectedOutput ?? "")),
        })
      : (input.expectedOutput as ZodSchema);

    const finalAnswerTool = new DynamicTool({
      name: "final_answer",
      description: "Sends the final answer to the user",
      inputSchema: finalAnswerToolSchema,
      handler: async (input) => {
        const result = usePlainResponse ? input.response : JSON.stringify(input);
        state.result = new AssistantMessage(result);
        return new StringToolOutput("Message has been sent");
      },
    });

    const tools = [...this.input.tools, finalAnswerTool];
    let forceFinalAnswer = false;

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
        tools,
        toolChoice: forceFinalAnswer ? finalAnswerTool : tools.length > 1 ? "required" : tools[0],
        stream: false,
      });
      await state.memory.addMany(response.messages);

      const toolCallMessages = response.getToolCalls();
      for (const toolCall of toolCallMessages) {
        try {
          const tool = tools.find((tool) => tool.name === toolCall.toolName);
          if (!tool) {
            throw new AgentError(`Tool ${toolCall.toolName} does not exist!`);
          }

          const toolInput: any = toolCall.input;
          const toolResponse: ToolOutput = await tool.run(toolInput).context({
            state,
            toolCallMsg: toolCall,
          });
          await state.memory.add(
            new ToolMessage({
              type: "tool-result",
              toolCallId: toolCall.toolCallId,
              toolName: toolCall.toolName,
              output: { type: "text", value: toolResponse.getTextContent() },
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
                output: { type: "error-text", value: e.explain() },
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

      // Fallback for providers that do not support structured outputs
      if (!isEmpty(textMessages) && isEmpty(toolCallMessages)) {
        forceFinalAnswer = true;
        tools.length = 0;
        tools.push(finalAnswerTool);
      }

      await run.emitter.emit("success", { state });
    }

    if (this.input.saveIntermediateSteps) {
      this.memory.reset();
      await this.memory.addMany(state.memory.messages.slice(1));
    } else {
      await this.memory.addMany(state.memory.messages.slice(-2));
    }
    return { memory: state.memory, result: state.result };
  }

  get meta(): AgentMeta {
    return {
      name: this.input.name || "RequirementAgent",
      tools: this.input.tools.slice(),
      description:
        this.input.description || "RequirementAgent that uses tools to accomplish the task.",
    };
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
