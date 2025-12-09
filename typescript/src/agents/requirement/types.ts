/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage, Message } from "@/backend/message.js";
import { Callback } from "@/emitter/types.js";
import {
  RequirementAgentSystemPrompt,
  RequirementAgentTaskPrompt,
} from "@/agents/requirement/prompts.js";
import { ZodSchema } from "zod";

export interface RequirementAgentRunInput {
  prompt?: string | Message[];
  context?: string;
  expectedOutput?: string | ZodSchema;
}

export interface RequirementAgentRunOutput {
  result: AssistantMessage;
  memory: BaseMemory;
}

export interface RequirementAgentRunState {
  result?: AssistantMessage;
  memory: BaseMemory;
  iteration: number;
}

export interface RequirementAgentExecutionConfig {
  totalMaxRetries?: number;
  maxRetriesPerStep?: number;
  maxIterations?: number;
}

export interface RequirementAgentRunOptions {
  signal?: AbortSignal;
  execution?: RequirementAgentExecutionConfig;
}

export interface RequirementAgentCallbacks {
  start?: Callback<{ state: RequirementAgentRunState }>;
  success?: Callback<{ state: RequirementAgentRunState }>;
  final_answer?: Callback<{
    state: RequirementAgentRunState;
    outputStructured: any;
    output: string;
    delta: string;
  }>;
}

export interface RequirementAgentTemplates {
  system: typeof RequirementAgentSystemPrompt;
  task: typeof RequirementAgentTaskPrompt;
}
