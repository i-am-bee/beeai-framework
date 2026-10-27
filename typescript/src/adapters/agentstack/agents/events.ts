/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Callback } from "@/emitter/types.js";

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackUpdateEvent {
  key: string;
  value: any;
}

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackErrorEvent {
  message: string;
}

/** @deprecated AgentStack is no longer maintained; use the A2AAgent or ACPAgent equivalents instead. */
export interface AgentStackAgentEvents {
  update?: Callback<AgentStackUpdateEvent>;
  error?: Callback<AgentStackErrorEvent>;
}
