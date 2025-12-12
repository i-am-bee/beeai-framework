/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage } from "@/backend/message.js";
import { AnyTool, ToolOutput } from "@/tools/base.js";
import { FrameworkError } from "@/errors.js";
import { PromptTemplate } from "@/template.js";
import { ChatModelOutput, ChatModelToolChoice, ChatModelUsage } from "@/backend/chat.js";
import { BaseAgentRunOptions } from "@/agents/base.js";
import { z } from "zod";
import { FinalAnswerTool } from "@/agents/requirement/utils/tool.js";

export const RequirementAgentSystemPromptInputSchema = z.object({
  role: z.string().optional(),
  instructions: z.string().optional(),
  notes: z.string().optional(),
  finalAnswerName: z.string(),
  finalAnswerSchema: z.string().optional(),
  finalAnswerInstructions: z.string().optional(),
  tools: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
      inputSchema: z.string(),
      allowed: z.string(),
      reason: z.string().optional(),
    }),
  ),
});

export const RequirementAgentTaskPromptInputSchema = z.object({
  prompt: z.string(),
  context: z.string().optional(),
  expectedOutput: z.string().optional(),
});

export const RequirementAgentToolErrorPromptInputSchema = z.object({
  reason: z.string(),
});

export const RequirementAgentToolNoResultPromptInputSchema = z.object({
  toolCall: z.object({
    tool: z.any().optional(),
    input: z.unknown(),
  }),
});

// Template collection
export interface RequirementAgentTemplates {
  system: PromptTemplate<typeof RequirementAgentSystemPromptInputSchema>;
  task: PromptTemplate<typeof RequirementAgentTaskPromptInputSchema>;
  toolError: PromptTemplate<typeof RequirementAgentToolErrorPromptInputSchema>;
  toolNoResult: PromptTemplate<typeof RequirementAgentToolNoResultPromptInputSchema>;
}

// Run state types
export interface RequirementAgentRunStateStep {
  id: string;
  iteration: number;
  tool: AnyTool | null;
  input: unknown;
  output: ToolOutput;
  error: FrameworkError | null;
}

export interface RequirementAgentRunState {
  answer: AssistantMessage | null;
  result: unknown;
  memory: BaseMemory;
  iteration: number;
  steps: RequirementAgentRunStateStep[];
  usage: ChatModelUsage;
  // cost: ChatModelCost; TODO: not supported yet
}

// Agent output
export interface RequirementAgentOutput {
  result: AssistantMessage;
  memory: BaseMemory;
  state: RequirementAgentRunState;
}

// Internal request structure
export interface RequirementAgentRequest {
  tools: AnyTool[];
  allowedTools: AnyTool[];
  reasonByTool: WeakMap<AnyTool, string | undefined>;
  hiddenTools: AnyTool[];
  toolChoice: ChatModelToolChoice;
  finalAnswer: FinalAnswerTool;
  canStop: boolean;
}

// Execution configuration
export interface RequirementAgentExecutionConfig {
  maxRetriesPerStep?: number;
  totalMaxRetries?: number;
  maxIterations?: number;
}

// Run input and options
export interface RequirementAgentRunInput {
  prompt: string | null;
  context?: string;
  expectedOutput?: string | z.ZodSchema;
}

export interface RequirementAgentRunOptions extends BaseAgentRunOptions {
  execution?: RequirementAgentExecutionConfig;
}

// Event callback types
export interface RequirementAgentCallbacks {
  start: (data: { state: RequirementAgentRunState; request: RequirementAgentRequest }) => void;
  success: (data: { state: RequirementAgentRunState; response: ChatModelOutput }) => void;
  finalAnswer: (data: {
    state: RequirementAgentRunState;
    output: string;
    delta: string;
    outputStructured: unknown;
  }) => void;
}
