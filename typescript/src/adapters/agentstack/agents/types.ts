/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackAgentRunInput {
  input: Message | string | Message[] | string[];
}

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackAgentRunOutput {
  result: Message;
  event: Record<string, any>;
}

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackAgentInput {
  url: string;
  agentName: string;
  memory: BaseMemory;
}
