/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyVoid } from "@/internals/types.js";
import { Emitter } from "@/emitter/emitter.js";

export type MatcherFn = (event: EventMeta) => boolean;
export type Matcher = "*" | "*.*" | RegExp | MatcherFn;
//export type Callback<T> = ((value: T) => AnyVoid) | ((value: T, event: EventMeta) => AnyVoid);
export type InferCallbackValue<T> = NonNullable<T> extends Callback<infer P> ? P : never;
export type Callback<T> = (value: T, event: EventMeta) => AnyVoid;
export type CleanupFn = () => void;
export type StringKey<T> = Extract<keyof T, string>;
export interface EmitterOptions {
  isBlocking?: boolean;
  once?: boolean;
  persistent?: boolean;
  matchNested?: boolean;
}
export interface EventTrace {
  id: string;
  runId: string;
  parentRunId?: string;
}
export interface EventMeta<C = unknown> {
  id: string;
  groupId?: string;
  name: string;
  path: string;
  createdAt: Date;
  source: Emitter<any>;
  creator: object;
  context: C;
  trace?: EventTrace;
}
