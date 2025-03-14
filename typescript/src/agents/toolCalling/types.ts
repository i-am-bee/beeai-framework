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
