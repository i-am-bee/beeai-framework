/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyTool } from "@/tools/base.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { FrameworkError, ValueError } from "@/errors.js";
import { toCamelCase } from "remeda";
import type { RequirementAgentRunState } from "@/agents/requirement/types.js";

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
  public middlewares: any[] = [];

  protected _priority = 10;
  protected _emitter?: Emitter<any>;

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

  get emitter(): Emitter<any> {
    if (!this._emitter) {
      this._emitter = this.createEmitter();
    }
    return this._emitter;
  }

  protected createEmitter(): Emitter<any> {
    return Emitter.root.child({
      namespace: ["requirement", toCamelCase(this.name)],
      creator: this,
    });
  }

  abstract run(state: RequirementAgentRunState): Promise<Rule[]>;

  async init(tools: AnyTool[], _: RunContext<any>): Promise<void> {
    await this.emitter.emit("init", { tools });
  }

  async clone(): Promise<this> {
    const instance = Object.create(Object.getPrototypeOf(this));
    instance.name = this.name;
    instance._priority = this._priority;
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
