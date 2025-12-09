// Copyright 2025 © BeeAI a Series of LF Projects, LLC
// SPDX-License-Identifier: Apache-2.0

import { v4 as uuidv4 } from "uuid";
import { AgentError, AgentExecutionConfig } from "@/agents.js";
import {
  RequirementAgentFinalAnswerEvent,
  RequirementAgentStartEvent,
  RequirementAgentSuccessEvent,
} from "@/agents/requirement/events.js";
import { RequirementAgentToolErrorPromptInput } from "@/agents/requirement/prompts.js";
import { Requirement, Rule } from "@/agents/requirement/requirements.js";
import {
  RequirementAgentRequest,
  RequirementAgentRunState,
  RequirementAgentRunStateStep,
  RequirementAgentTemplates,
} from "@/agents/requirement/types.js";
import { RequirementsReasoner, createSystemMessage } from "@/agents/requirement/utils/llm.js";
import {
  FinalAnswerTool,
  FinalAnswerToolSchema,
  runTools,
} from "@/agents/requirement/utils/tool.js";
import { ToolCallChecker } from "@/agents/tool-calling/utils.js";
import {
  AnyMessage,
  AssistantMessage,
  ChatModel,
  ChatModelOutput,
  MessageToolCallContent,
  MessageToolResultContent,
  ToolMessage,
} from "@/backend.js";
import { ChatModelOptions } from "@/backend/chat.js";
import { parseBrokenJson } from "@/backend/utils.js";
import { RunContext } from "@/context.js";
import { UnconstrainedMemory } from "@/memory.js";
import { TEMP_MESSAGE_META_KEY, deleteMessagesByMetaKey } from "@/memory/utils.js";
import { StreamToolCallMiddleware } from "@/middleware/stream-tool-call.js";
import { AnyTool } from "@/tools.js";
import { RetryCounter } from "@/utils/counter.js";
import { ensureStrictlyIncreasing, findLastIndex } from "@/utils/lists.js";
import { findFirstPair, generateRandomString, toJson } from "@/utils/strings.js";

interface RequirementAgentRunnerConfig {
  config: AgentExecutionConfig;
  toolCallCycleChecker: ToolCallChecker;
  forceFinalAnswerAsTool: boolean;
  expectedOutput: any;
  runContext: RunContext;
  requirements: Requirement<RequirementAgentRunState>[];
  tools: AnyTool[];
  templates: RequirementAgentTemplates;
  llm: ChatModel;
}

export class RequirementAgentRunner {
  private readonly ctx: RunContext;
  private readonly llm: ChatModel;
  private readonly templates: RequirementAgentTemplates;
  private forceFinalAnswerAsTool: boolean;
  private readonly state: RequirementAgentRunState;
  private readonly requirements: Requirement<RequirementAgentRunState>[];
  private readonly reasoner: RequirementsReasoner;
  private readonly runConfig: AgentExecutionConfig;
  private readonly toolCallCycleChecker: ToolCallChecker;
  private readonly iterationErrorCounter: RetryCounter<AgentError>;
  private readonly globalErrorCounter: RetryCounter<AgentError>;

  constructor({
    config,
    toolCallCycleChecker,
    forceFinalAnswerAsTool,
    expectedOutput,
    runContext,
    requirements,
    tools,
    templates,
    llm,
  }: RequirementAgentRunnerConfig) {
    this.ctx = runContext;
    this.llm = llm;
    this.templates = templates;
    this.forceFinalAnswerAsTool = forceFinalAnswerAsTool;
    this.state = {
      answer: null,
      result: null,
      memory: new UnconstrainedMemory(),
      steps: [],
      iteration: 0,
    } as RequirementAgentRunState;
    this.requirements = requirements;
    this.reasoner = new RequirementsReasoner({
      tools,
      finalAnswer: new FinalAnswerTool(expectedOutput, this.state),
      context: runContext,
    });
    this.runConfig = config;
    this.toolCallCycleChecker = toolCallCycleChecker;

    const maxRetriesPerIteration = config.maxRetriesPerStep ?? 0;
    this.iterationErrorCounter = new RetryCounter({
      errorType: AgentError,
      maxRetries: maxRetriesPerIteration,
    });

    const maxRetries = Math.max(maxRetriesPerIteration, config.totalMaxRetries ?? 0);
    this.globalErrorCounter = new RetryCounter({
      errorType: AgentError,
      maxRetries,
    });
  }

  private incrementIteration(): void {
    this.state.iteration++;

    if (this.runConfig.maxIterations && this.state.iteration > this.runConfig.maxIterations) {
      throw new AgentError(
        `Agent was not able to resolve the task in ${this.state.iteration} iterations.`,
      );
    }
  }

  private createFinalAnswerStream(finalAnswerTool: FinalAnswerTool): StreamToolCallMiddleware {
    const streamMiddleware = new StreamToolCallMiddleware({
      tool: finalAnswerTool,
      field: "response", // from the default schema
      matchNested: false,
      forceStreaming: false,
    });

    streamMiddleware.emitter.on("update", (data, meta) => {
      this.ctx.emitter.emit(
        "final_answer",
        new RequirementAgentFinalAnswerEvent({
          state: this.state,
          output: data.output,
          delta: data.delta,
          outputStructured: null,
        }),
      );
    });

    return streamMiddleware;
  }

  private async runLlm(request: RequirementAgentRequest): Promise<ChatModelOutput> {
    const streamMiddleware = this.createFinalAnswerStream(request.finalAnswer);
    const [messages, options] = this.prepareLlmRequest(request);
    const response = await this.llm.run(messages, options).middleware(streamMiddleware);

    this.state.usage.merge(response.usage);
    this.state.cost.merge(response.cost);

    streamMiddleware.unbind();
    return response;
  }

  private prepareLlmRequest(request: RequirementAgentRequest): [AnyMessage[], ChatModelOptions] {
    const messages: AnyMessage[] = [
      createSystemMessage({
        template: this.templates.system,
        request,
      }),
      ...this.state.memory.messages,
    ];

    const options: ChatModelOptions = {
      maxRetries: this.runConfig.maxRetriesPerStep,
      tools: request.allowedTools,
      toolChoice: request.toolChoice,
      streamPartialToolCalls: true,
    };

    const cacheControlInjectionPoints = [
      {
        location: "message" as const,
        index: this.requirements.length > 0 ? 1 : 0, // system prompt might be dynamic when requirements are set
      },
      {
        location: "message" as const,
        index: findLastIndex(
          messages,
          (msg) =>
            !msg.meta?.[TEMP_MESSAGE_META_KEY] &&
            // TODO: remove once https://github.com/BerriAI/litellm/issues/17479 is resolved
            (this.llm.providerId !== "amazon_bedrock" || !(msg instanceof ToolMessage)),
        ),
      },
    ];

    options.cacheControlInjectionPoints = ensureStrictlyIncreasing(
      cacheControlInjectionPoints,
      (v) => v.index, // prevent duplicates
    );

    return [messages, options];
  }

  private async createFinalAnswerToolCall(fullText: string): Promise<AssistantMessage | null> {
    /**Try to convert a text message to a valid final answer tool call.*/
    const jsonObjectPair = findFirstPair(fullText, ["{", "}"]);
    let finalAnswerInput = jsonObjectPair ? parseBrokenJson(jsonObjectPair.outer) : null;

    if (!finalAnswerInput && !this.reasoner.finalAnswer.customSchema) {
      finalAnswerInput = new FinalAnswerToolSchema({
        response: fullText,
      }).toObject();
    }

    if (!finalAnswerInput) {
      return null;
    }

    const manualAssistantToolCallMessage: MessageToolCallContent = {
      type: "tool-call",
      id: `call_${generateRandomString(8).toLowerCase()}`,
      toolName: this.reasoner.finalAnswer.name,
      args: toJson(finalAnswerInput, { sortKeys: false }),
    };

    return new AssistantMessage(manualAssistantToolCallMessage);
  }

  private async createRequest(options?: { extraRules?: Rule[] }): Promise<RequirementAgentRequest> {
    return await this.reasoner.createRequest({
      state: this.state,
      forceToolCall: this.forceFinalAnswerAsTool,
      extraRules: options?.extraRules,
    });
  }

  private async invokeToolCalls(
    tools: AnyTool[],
    toolCalls: MessageToolCallContent[],
  ): Promise<ToolMessage[]> {
    const toolResults: ToolMessage[] = [];

    const toolCallResults = await runTools({
      tools,
      messages: toolCalls,
      context: { state: this.state.toObject() },
    });

    for (const toolCall of toolCallResults) {
      this.state.steps.push({
        id: uuidv4(),
        iteration: this.state.iteration,
        input: toolCall.input,
        output: toolCall.output,
        tool: toolCall.tool,
        error: toolCall.error,
      } as RequirementAgentRunStateStep);

      let result: string;
      if (toolCall.error !== null) {
        result = this.templates.toolError.render(
          new RequirementAgentToolErrorPromptInput({
            reason: toolCall.error.explain(),
          }),
        );
      } else {
        result = !toolCall.output.isEmpty()
          ? toolCall.output.getTextContent()
          : this.templates.toolNoResult.render({ toolCall });
      }

      toolResults.push(
        new ToolMessage({
          type: "tool-result",
          toolName: toolCall.tool?.name ?? toolCall.msg.toolName,
          toolCallId: toolCall.msg.id,
          result,
        } as MessageToolResultContent),
      );

      if (toolCall.error !== null) {
        this.iterationErrorCounter.use(toolCall.error);
        this.globalErrorCounter.use(toolCall.error);
      }
    }

    return toolResults;
  }

  public async addMessages(messages: AnyMessage[]): Promise<void> {
    await this.state.memory.addMany(messages);
  }

  public async run(): Promise<RequirementAgentRunState> {
    /**Run the agent until it reaches the final answer. Returns the final state.*/
    if (this.state.answer !== null) {
      return this.state;
    }

    // Init requirements
    await this.reasoner.update(this.requirements);

    while (this.state.answer === null) {
      this.incrementIteration();

      const request = await this.createRequest();
      await this.ctx.emitter.emit(
        "start",
        new RequirementAgentStartEvent({ state: this.state, request }),
      );

      this.iterationErrorCounter.reset();
      const response = await this.runSingle(request);

      await this.ctx.emitter.emit(
        "success",
        new RequirementAgentSuccessEvent({ state: this.state, response }),
      );
    }

    return this.state;
  }

  private async runSingle(request: RequirementAgentRequest): Promise<ChatModelOutput> {
    /**Run a single iteration of the agent.*/
    const response = await this.runLlm(request);

    // Try to cast a text message to a final answer tool call if it is allowed
    if (!response.getToolCalls()?.length) {
      const text = response.getTextContent();
      if (!text || request.canStop) {
        throw new AgentError("Model produced an empty response.", {
          context: { response },
        });
      }

      const finalAnswerToolCall = await this.createFinalAnswerToolCall(text);
      if (!finalAnswerToolCall) {
        const err = new AgentError("Model produced an invalid final answer tool call.");
        this.iterationErrorCounter.use(err);
        this.globalErrorCounter.use(err);

        await this.reasoner.update([]);
        const updatedRequest = await this.createRequest({
          extraRules: [
            new Rule({
              target: this.reasoner.finalAnswer.name,
              allowed: true,
              hidden: false,
            }),
          ],
        });
        this.forceFinalAnswerAsTool = true;
        return await this.runSingle(updatedRequest);
      }

      response.outputStructured = null;
      response.output = [finalAnswerToolCall];
    }

    // Check for cycles
    const toolCalls = response.getToolCalls();
    for (const toolCallMsg of toolCalls) {
      this.toolCallCycleChecker.register(toolCallMsg);
      if (this.toolCallCycleChecker.cycleFound) {
        this.toolCallCycleChecker.reset();
        const updatedRequest = await this.createRequest({
          extraRules: [
            new Rule({
              target: toolCallMsg.toolName,
              allowed: false,
              hidden: false,
              forced: true,
            }),
          ],
        });
        return await this.runSingle(updatedRequest);
      }
    }

    const toolResults = await this.invokeToolCalls(request.allowedTools, toolCalls);

    await this.state.memory.addMany([...response.output, ...toolResults]);
    await deleteMessagesByMetaKey(this.state.memory, {
      key: TEMP_MESSAGE_META_KEY,
      value: true,
    });

    return response;
  }
}
