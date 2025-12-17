/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyTool } from "@/tools/base.js";
import { MiddlewareType, RunContext } from "@/context.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { FrameworkError, ValueError } from "@/errors.js";
import { toCamelCase } from "remeda";
import type { RequirementAgentRunState } from "@/agents/requirement/types.js";
import { Cache } from "@/cache/decoratorCache.js";

// Rule definition
export interface Rule {
  target: string;
  allowed: boolean;
  reason?: string;
  preventStop: boolean;
  forced: boolean;
  hidden: boolean;
}

// Base Requirement class
export abstract class Requirement {
  public name: string;
  public state: Record<string, any> = {};
  public enabled = true;
  public middlewares: MiddlewareType<typeof this>[] = [];

  protected _priority = 10;

  constructor(name?: string) {
    this.name = name || this.constructor.name;
  }

  get priority(): number {
    return this._priority;
  }

  set priority(value: number) {
    if (value <= 0) {
      throw new ValueError("Priority must be a positive integer.");
    }
    this._priority = value;
  }

  @Cache({ enumerable: false })
  get emitter(): Emitter<RequirementCallbacks> {
    return Emitter.root.child({
      namespace: ["requirement", toCamelCase(this.name)],
      creator: this,
    });
  }

  run(state: RequirementAgentRunState) {
    return RunContext.enter(
      this,
      { signal: undefined, params: [state] as const },
      async (context) => {
        return await this._run(state, context);
      },
    );
  }

  abstract _run(state: RequirementAgentRunState, _: RunContext<typeof this>): Promise<Rule[]>;

  async init(tools: AnyTool[], _: RunContext<any>): Promise<void> {
    await this.emitter.emit("init", { tools });
  }

  async clone(): Promise<this> {
    const instance = Object.create(Object.getPrototypeOf(this));
    instance.name = this.name;
    instance.priority = this.priority;
    instance.enabled = this.enabled;
    instance.state = { ...this.state };
    instance.middlewares = [...this.middlewares];
    return instance;
  }
}

// Requirement error
export class RequirementError extends FrameworkError {
  constructor(
    message: string,
    public readonly requirement?: Requirement,
    cause?: Error,
    context?: Record<string, any>,
  ) {
    super(message, cause ? [cause] : undefined, {
      isFatal: true,
      isRetryable: false,
      context,
    });
  }
}

export interface RequirementCallbacks {
  init: Callback<{ tools: AnyTool[] }>;
}
