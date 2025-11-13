/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";

export interface AgentStackAgentRunInput {
  input: Message | string | Message[] | string[];
}

export interface AgentStackAgentRunOutput {
  result: Message;
  event: Record<string, any>;
}

export interface AgentStackAgentInput {
  url: string;
  agentName: string;
  memory: BaseMemory;
}
