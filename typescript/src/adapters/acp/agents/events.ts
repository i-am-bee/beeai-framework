/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Callback } from "@/emitter/types.js";

export interface ACPAgentUpdateEvent {
  key: string;
  value: any;
}

export interface ACPAgentErrorEvent {
  message: string;
}

export interface ACPAgentEvents {
  update?: Callback<ACPAgentUpdateEvent>;
  error?: Callback<ACPAgentErrorEvent>;
}
