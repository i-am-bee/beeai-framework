/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Callback } from "@/emitter/types.js";

export interface A2AAgentUpdateEvent {
  value: any;
}

export interface A2AAgentErrorEvent {
  message: string;
}

export interface A2AAgentEvents {
  update?: Callback<A2AAgentUpdateEvent>;
  error?: Callback<A2AAgentErrorEvent>;
}
