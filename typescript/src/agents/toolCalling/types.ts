/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage } from "@/backend/message.js";
import { Callback } from "@/emitter/types.js";
import {
  ToolCallingAgentSystemPrompt,
  ToolCallingAgentTaskPrompt,
} from "@/agents/toolCalling/prompts.js";
import { ZodSchema } from "zod";

export interface ToolCallingAgentRunInput {
  prompt?: string;
  context?: string;
  expectedOutput?: string | ZodSchema;
}

export interface ToolCallingAgentRunOutput {
  result: AssistantMessage;
  memory: BaseMemory;
}

export interface ToolCallingAgentRunState {
  result?: AssistantMessage;
  memory: BaseMemory;
  iteration: number;
}

export interface ToolCallingAgentExecutionConfig {
  totalMaxRetries?: number;
  maxRetriesPerStep?: number;
  maxIterations?: number;
}

export interface ToolCallingAgentRunOptions {
  signal?: AbortSignal;
  execution?: ToolCallingAgentExecutionConfig;
}

export interface ToolCallingAgentCallbacks {
  start?: Callback<{ state: ToolCallingAgentRunState }>;
  success?: Callback<{ state: ToolCallingAgentRunState }>;
}

export interface ToolCallingAgentTemplates {
  system: typeof ToolCallingAgentSystemPrompt;
  task: typeof ToolCallingAgentTaskPrompt;
}
