/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { BaseAgentRunOptions } from "@/agents/base.js";
import { AgentCard, Message as A2AMessage, StreamResponse } from "@a2a-js/sdk";

export interface A2AAgentRunInput {
  input: string | Message | Message[] | A2AMessage;
}

export interface A2AAgentRunOutput {
  result: Message;
  event: StreamResponse;
}

export interface A2AAgentInput {
  url?: string;
  agentCardPath?: string;
  agentCard?: AgentCard;
  memory: BaseMemory;
}

export interface A2AAgentRunOptions extends BaseAgentRunOptions {
  contextId?: string;
  taskId?: string;
  clearContext?: boolean;
}
