/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Callback } from "@/emitter/types.js";

export interface BeeAIPlatformUpdateEvent {
  key: string;
  value: any;
}

export interface BeeAIPlatformErrorEvent {
  message: string;
}

export interface BeeAIPlatformAgentEvents {
  update?: Callback<BeeAIPlatformUpdateEvent>;
  error?: Callback<BeeAIPlatformErrorEvent>;
}
