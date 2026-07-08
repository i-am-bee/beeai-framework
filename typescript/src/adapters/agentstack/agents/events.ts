/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Callback } from "@/emitter/types.js";

export interface AgentStackUpdateEvent {
  key: string;
  value: any;
}

export interface AgentStackErrorEvent {
  message: string;
}

export interface AgentStackAgentEvents {
  update?: Callback<AgentStackUpdateEvent>;
  error?: Callback<AgentStackErrorEvent>;
}
