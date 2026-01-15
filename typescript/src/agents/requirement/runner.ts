/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { v4 as uuidv4 } from "uuid";
import { AgentError } from "@/agents/base.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, Message, ToolMessage } from "@/backend/message.js";
import { ToolCallPart } from "ai";
import { AnyTool } from "@/tools/base.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { RetryCounter } from "@/internals/helpers/counter.js";
import { RunContext } from "@/context.js";
import {
  RequirementAgentExecutionConfig,
  RequirementAgentRequest,
  RequirementAgentRunState,
  RequirementAgentTemplates,
} from "./types.js";
import { Requirement, Rule } from "./requirements/requirement.js";
import { RequirementsReasoner, createSystemMessage } from "./utils/llm.js";
import { FinalAnswerTool, runTool } from "./utils/tool.js";
import { ToolCallChecker } from "./utils/toolCallChecker.js";
import { z } from "zod";
import { parseBrokenJson } from "@/internals/helpers/schema.js";
import { RequirementAgent } from "@/agents/requirement/agent.js";
import { mergeTokenUsage } from "@/adapters/vercel/backend/utils.js";
import { StreamToolCallMiddleware } from "@/middleware/streamToolCall.js";

const TEMP_MESSAGE_KEY = "tempMessage";

/**
 * Runner for RequirementAgent execution
 */
export class RequirementAgentRunner {
  protected readonly state: RequirementAgentRunState;
  protected readonly reasoner: RequirementsReasoner;
  protected readonly iterationErrorCounter: RetryCounter;
  protected readonly globalErrorCounter: RetryCounter;

  constructor(
    protected readonly llm: ChatModel,
    protected readonly runConfig: RequirementAgentExecutionConfig,
    tools: AnyTool[],
    protected readonly requirements: Requirement[],
    expectedOutput: string | z.ZodSchema | null,
    protected readonly toolCallCycleChecker: ToolCallChecker,
    protected readonly ctx: RunContext<RequirementAgent>,
    protected forceFinalAnswerAsTool: boolean,
    protected readonly templates: RequirementAgentTemplates,
  ) {
    this.state = {
      answer: null,
      result: null,
      memory: new UnconstrainedMemory(),
      steps: [],
      iteration: 0,
      usage: {
        totalTokens: 0,
        promptTokens: 0,
        completionTokens: 0,
        reasoningTokens: 0,
        cachedPromptTokens: undefined,
      },
    };

    this.requirements = requirements;
    const finalAnswer = new FinalAnswerTool(expectedOutput, this.state);
    this.reasoner = new RequirementsReasoner(tools, finalAnswer, ctx);

    const maxRetriesPerIteration = runConfig.maxRetriesPerStep ?? 0;
    this.iterationErrorCounter = new RetryCounter(maxRetriesPerIteration, AgentError);

    const maxRetries = Math.max(maxRetriesPerIteration, runConfig.totalMaxRetries ?? 0);
    this.globalErrorCounter = new RetryCounter(maxRetries, AgentError);
  }

  protected incrementIteration(): void {
    this.state.iteration++;

    if (this.runConfig.maxIterations && this.state.iteration > this.runConfig.maxIterations) {
      throw new AgentError(
        `Agent was not able to resolve the task in ${this.state.iteration} iterations.`,
      );
    }
  }

  protected async runLLM(request: RequirementAgentRequest): Promise<ChatModelOutput> {
    const streamMiddleware = this.createFinalAnswerStream(request.finalAnswer);
    try {
      const input = await this.prepareLLMRequest(request);
      const response = await this.llm.create(input).middleware(streamMiddleware);

      if (response.usage) {
        mergeTokenUsage(this.state.usage, response.usage);
      }

      return response;
    } finally {
      streamMiddleware.unbind();
    }
  }

  protected async prepareLLMRequest(request: RequirementAgentRequest): Promise<ChatModelInput> {
    const messages: Message[] = [
      await createSystemMessage(this.templates.system, request),
      ...this.state.memory.messages,
    ];

    return {
      messages,
      tools: request.allowedTools,
      toolChoice: request.toolChoice,
      streamPartialToolCalls: true,
      maxRetries: this.runConfig.maxRetriesPerStep ?? 0,
    };
  }

  protected async createFinalAnswerToolCall(fullText: string): Promise<AssistantMessage | null> {
    // Try to extract JSON from text
    const parsed = parseBrokenJson(fullText, { pair: ["{", "}"] });
    if (!parsed) {
      // If no JSON and no custom schema, wrap in default schema
      if (!this.reasoner.finalAnswer.customSchema) {
        return new AssistantMessage([
          {
            type: "tool-call",
            toolCallId: `call_${uuidv4().substring(0, 8)}`,
            toolName: this.reasoner.finalAnswer.name,
            input: this.reasoner.finalAnswer.inputSchema().parse({ response: fullText }),
          },
        ]);
      }
      return null;
    }

    return new AssistantMessage([
      {
        type: "tool-call",
        toolCallId: `call_${uuidv4().substring(0, 8)}`,
        toolName: this.reasoner.finalAnswer.name,
        input: parsed,
      },
    ]);
  }

  protected async createRequest(extraRules: Rule[] = []): Promise<RequirementAgentRequest> {
    return await this.reasoner.createRequest(this.state, this.forceFinalAnswerAsTool, extraRules);
  }

  protected async invokeToolCalls(
    tools: AnyTool[],
    toolCalls: ToolCallPart[],
  ): Promise<ToolMessage[]> {
    const toolResults: ToolMessage[] = [];

    const results = await Promise.all(
      toolCalls.map((msg) => runTool(tools, msg, { state: this.state })),
    );

    for (const toolCall of results) {
      this.state.steps.push({
        id: uuidv4(),
        iteration: this.state.iteration,
        input: toolCall.input,
        output: toolCall.output,
        tool: toolCall.tool,
        error: toolCall.error,
      });

      let result: string;
      if (toolCall.error) {
        result = this.templates.toolError.render({
          reason: toolCall.error.explain(),
        });
      } else {
        result = !toolCall.output.isEmpty()
          ? toolCall.output.getTextContent()
          : this.templates.toolNoResult.render({
              toolCall: { tool: toolCall.tool, input: toolCall.input },
            });
      }
      toolResults.push(
        new ToolMessage({
          type: "tool-result",
          toolName: toolCall.tool?.name || toolCall.msg.toolName,
          toolCallId: toolCall.msg.toolCallId,
          output: { type: "text", value: result },
        }),
      );

      if (toolCall.error) {
        this.iterationErrorCounter.use(toolCall.error);
        this.globalErrorCounter.use(toolCall.error);
      }
    }

    return toolResults;
  }

  async addMessages(messages: Message[]): Promise<void> {
    await this.state.memory.addMany(messages);
  }

  async run(): Promise<RequirementAgentRunState> {
    if (this.state.answer) {
      return this.state;
    }

    // Initialize requirements
    await this.reasoner.update(this.requirements);

    while (!this.state.answer) {
      this.incrementIteration();

      const request = await this.createRequest();
      await this.ctx.emitter.emit("start", { state: this.state, request });
      const response = await this.runIteration(request);
      await this.ctx.emitter.emit("success", { state: this.state, response });
    }

    return this.state;
  }

  protected async runIteration(request: RequirementAgentRequest): Promise<ChatModelOutput> {
    const response = await this.runLLM(request);

    // Try to cast text message to final answer tool call if allowed
    const toolCalls = response.getToolCalls();
    if (toolCalls.length === 0) {
      const textMessages = response.getTextMessages();
      const text = textMessages.map((m) => m.text).join("\n");

      const finalAnswerToolCall =
        text && request.canStop ? await this.createFinalAnswerToolCall(text) : null;
      if (finalAnswerToolCall) {
        const stream = this.createFinalAnswerStream(request.finalAnswer);
        await stream.add(new ChatModelOutput([finalAnswerToolCall]));
      } else {
        const err = new AgentError("Model produced an invalid final answer tool call.");
        this.iterationErrorCounter.use(err);
        this.globalErrorCounter.use(err);

        await this.reasoner.update([]);
        const updatedRequest = await this.createRequest([
          {
            target: this.reasoner.finalAnswer.name,
            allowed: true,
            hidden: false,
            forced: false,
            preventStop: false,
          },
        ]);
        this.forceFinalAnswerAsTool = true;
        return await this.runIteration(updatedRequest);
      }

      response.messages.length = 0;
      toolCalls.push(...finalAnswerToolCall.getToolCalls());
    }

    // Check for cycles
    for (const toolCallMsg of toolCalls) {
      this.toolCallCycleChecker.register(toolCallMsg);
      if (this.toolCallCycleChecker.cycleFound) {
        this.toolCallCycleChecker.reset();
        const updatedRequest = await this.createRequest([
          {
            target: toolCallMsg.toolName,
            allowed: false,
            hidden: false,
            forced: true,
            preventStop: false,
          },
        ]);
        return await this.runIteration(updatedRequest);
      }
    }

    const toolResults = await this.invokeToolCalls(request.allowedTools, toolCalls);
    await this.state.memory.addMany([...response.messages, ...toolResults]);

    // Delete temporary messages
    const tempMessages = this.state.memory.messages.filter((msg) => msg.meta[TEMP_MESSAGE_KEY]);
    await this.state.memory.deleteMany(tempMessages);

    return response;
  }

  private createFinalAnswerStream(finalAnswer: FinalAnswerTool) {
    const middleware = new StreamToolCallMiddleware({
      target: finalAnswer,
      forceStreaming: false,
      key: "response",
      matchNested: false,
    });
    middleware.emitter.on("update", async (data, _) => {
      await this.ctx.emitter.emit("finalAnswer", {
        state: this.state,
        output: data.output,
        delta: data.delta,
        outputStructured: undefined,
      });
    });
    return middleware;
  }
}
