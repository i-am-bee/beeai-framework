/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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
