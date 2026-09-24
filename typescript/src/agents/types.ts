/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseAgent } from "@/agents/base.js";
import { OmitEmpty } from "@/internals/types.js";
import { AnyTool } from "@/tools/base.js";

export interface AgentMeta {
  name: string;
  description: string;
  extraDescription?: string;
  tools: AnyTool[];
}

export type AgentCallbackValue =
  | { data?: never; error: Error }
  | { data: unknown; error?: never }
  | object;

export type InternalAgentCallbackValue<
  T extends AgentCallbackValue,
  E extends NonNullable<unknown>,
> = OmitEmpty<T> & E;

export type PublicAgentCallbackValue<T extends AgentCallbackValue = AgentCallbackValue> =
  OmitEmpty<T>;

export type AgentCallback<T extends PublicAgentCallbackValue> = (value: T) => void;

export type GetAgentInput<T> = T extends BaseAgent<infer X, any, any> ? X : never;
export type GetAgentOutput<T> = T extends BaseAgent<any, infer X, any> ? X : never;
export type AnyAgent = BaseAgent<any, any, any>;
