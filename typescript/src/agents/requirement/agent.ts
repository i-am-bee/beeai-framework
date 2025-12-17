/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseAgent } from "@/agents/base.js";
import { AnyTool } from "@/tools/base.js";
import { BaseMemory } from "@/memory/base.js";
import { AgentMeta } from "@/agents/types.js";
import { Emitter } from "@/emitter/emitter.js";
import { ChatModel } from "@/backend/chat.js";
import type { GetRunContext, MiddlewareType } from "@/context.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { Message, UserMessage } from "@/backend/message.js";
import { shallowCopy } from "@/serializer/utils.js";
import { mapObj } from "@/internals/helpers/object.js";
import { PromptTemplate } from "@/template.js";
import {
  RequirementAgentCallbacks,
  RequirementAgentExecutionConfig,
  RequirementAgentOutput,
  RequirementAgentRunInput,
  RequirementAgentRunOptions,
  RequirementAgentTemplates,
} from "./types.js";
import {
  RequirementAgentSystemPrompt,
  RequirementAgentTaskPrompt,
  RequirementAgentToolErrorPrompt,
  RequirementAgentToolNoResultPrompt,
} from "./prompts.js";
import { Requirement } from "./requirements/requirement.js";
import { RequirementAgentRunner } from "./runner.js";
import { ToolCallChecker, ToolCallCheckerConfig } from "./utils/toolCallChecker.js";
import { isDefined, isEmptyish, pickBy } from "remeda";
import { castArray } from "@/internals/helpers/array.js";
import type { RequiredExcept } from "@/internals/types.js";

export type RequirementAgentTemplateFactory<K extends keyof RequirementAgentTemplates> = (
  template: RequirementAgentTemplates[K],
) => RequirementAgentTemplates[K];

export interface RequirementAgentInput {
  llm: ChatModel;
  memory?: BaseMemory;
  tools?: AnyTool[];
  requirements?: Requirement[];
  name?: string;
  description?: string;
  role?: string;
  instructions?: string | string[];
  notes?: string | string[];
  toolCallChecker?: boolean | ToolCallCheckerConfig;
  finalAnswerAsTool?: boolean;
  saveIntermediateSteps?: boolean;
  execution?: RequirementAgentExecutionConfig;
  templates?: Partial<{
    [K in keyof RequirementAgentTemplates]:
      | RequirementAgentTemplates[K]
      | RequirementAgentTemplateFactory<K>;
  }>;
  middlewares?: MiddlewareType<RequirementAgent>[];
}

type InternalRequirementAgentInput = RequiredExcept<
  RequirementAgentInput,
  | "role"
  | "instructions"
  | "name"
  | "description"
  | "templates"
  | "notes"
  | "execution"
  | "middlewares"
>;

/**
 * RequirementAgent - A declarative AI agent with rule-based constraints
 *
 * The RequirementAgent provides predictable, controlled execution behavior across
 * different language models through rule-based constraints (Requirements).
 * Requirements can be configured as strict or flexible as necessary, adapting to
 * task requirements while ensuring consistent execution regardless of the underlying
 * model's reasoning or tool-calling capabilities.
 */
export class RequirementAgent extends BaseAgent<
  RequirementAgentRunInput,
  RequirementAgentOutput,
  RequirementAgentRunOptions
> {
  public readonly emitter = Emitter.root.child<RequirementAgentCallbacks>({
    namespace: ["agent", "requirement"],
    creator: this,
  });

  protected readonly input: InternalRequirementAgentInput;

  protected runner: new (
    ...args: ConstructorParameters<typeof RequirementAgentRunner>
  ) => RequirementAgentRunner = RequirementAgentRunner;

  constructor(input: RequirementAgentInput) {
    super();
    this.input = {
      ...input,
      memory: input.memory || new UnconstrainedMemory(),
      tools: input.tools || [],
      requirements: input.requirements || [],
      saveIntermediateSteps: input.saveIntermediateSteps ?? true,
      finalAnswerAsTool: input.finalAnswerAsTool ?? true,
      toolCallChecker: input.toolCallChecker ?? true,
    };
    if (input.middlewares) {
      this.middlewares.push(...input.middlewares);
    }
  }

  static {
    this.register();
  }

  protected async _run(
    input: RequirementAgentRunInput,
    options: RequirementAgentRunOptions = {},
    run: GetRunContext<typeof this>,
  ): Promise<RequirementAgentOutput> {
    const execution: RequirementAgentExecutionConfig = {
      maxRetriesPerStep: 3,
      totalMaxRetries: 20,
      maxIterations: 20,
      ...this.input.execution,
      ...options.execution,
    };

    const toolCallChecker = this.createToolCallChecker();

    const runner = new this.runner(
      this.input.llm,
      execution,
      this.input.tools || [],
      this.input.requirements || [],
      input.expectedOutput || null,
      toolCallChecker,
      run,
      this.input.finalAnswerAsTool,
      this.templates,
    );

    // Process input messages
    const newMessages = this.processInput(input);
    await runner.addMessages([...this.memory.messages]);
    await runner.addMessages(newMessages);

    // Run the agent
    const finalState = await runner.run();

    // Update memory
    if (this.input.saveIntermediateSteps) {
      this.memory.reset();
      await this.memory.addMany(finalState.memory.messages);
    } else {
      await this.memory.addMany(newMessages);
      // Add last tool call pair
      const messages = finalState.memory.messages;
      const lastPair = messages.slice(-2);
      await this.memory.addMany(lastPair);
    }

    if (!finalState.answer) {
      throw new Error("Agent did not produce a final answer");
    }

    return {
      result: finalState.answer,
      memory: finalState.memory,
      state: finalState,
    };
  }

  protected processInput(input: RequirementAgentRunInput): Message[] {
    if (!input.prompt) {
      return [];
    }

    const userMessage = new UserMessage(
      this.templates.task.render({
        prompt: input.prompt,
        context: input.context,
        expectedOutput: typeof input.expectedOutput === "string" ? input.expectedOutput : undefined,
      }),
    );

    return [userMessage];
  }

  protected createToolCallChecker(): ToolCallChecker {
    const config: ToolCallCheckerConfig = {};

    if (typeof this.input.toolCallChecker === "object") {
      Object.assign(config, this.input.toolCallChecker);
    }

    const checker = new ToolCallChecker(config);
    checker.enabled = this.input.toolCallChecker !== false;
    return checker;
  }

  get meta(): AgentMeta {
    const tools = this.input.tools || [];

    return {
      name: this.input.name ?? "Requirement",
      tools,
      description:
        this.input.description ??
        "The RequirementAgent is a declarative AI agent implementation that provides predictable, " +
          "controlled execution behavior across different language models through rule-based constraints.",
      ...(tools.length > 0 && {
        extraDescription: [
          `Tools that I can use to accomplish given task.`,
          ...tools.map((tool) => `Tool '${tool.name}': ${tool.description}.`),
        ].join("\n"),
      }),
    };
  }

  protected get templates(): RequirementAgentTemplates {
    const overrides = this.input.templates || {};
    const finalized = mapObj({
      system: RequirementAgentSystemPrompt,
      task: RequirementAgentTaskPrompt,
      toolError: RequirementAgentToolErrorPrompt,
      toolNoResult: RequirementAgentToolNoResultPrompt,
    } as RequirementAgentTemplates)((
      key,
      defaultTemplate: RequirementAgentTemplates[typeof key],
    ) => {
      const override = overrides[key] ?? defaultTemplate;
      if (override instanceof PromptTemplate) {
        return override;
      }
      return override(defaultTemplate) ?? defaultTemplate;
    });

    if (this.input.role || this.input.instructions || this.input.notes) {
      finalized.system.update({
        defaults: pickBy(
          {
            role: this.input.role,
            instructions: isEmptyish(this.input.instructions)
              ? undefined
              : castArray(this.input.instructions)
                  .map((v) => `- ${v}`)
                  .join("\n"),
            notes: isEmptyish(this.input.notes)
              ? undefined
              : castArray(this.input.notes)
                  .map((v) => `- ${v}`)
                  .join("\n"),
          },
          isDefined,
        ),
      });
    }

    return finalized;
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      emitter: this.emitter,
      runnerClass: this.runner,
    };
  }

  set memory(memory: BaseMemory) {
    this.input.memory = memory;
  }

  get memory(): BaseMemory {
    return this.input.memory!;
  }
}
