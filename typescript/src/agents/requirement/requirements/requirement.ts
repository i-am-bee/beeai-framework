// Copyright 2025 © BeeAI a Series of LF Projects, LLC
// SPDX-License-Identifier: Apache-2.0

import { MultiTargetType, assertAllRulesFound, extractTargets } from "./requirements/_utils";
import { RequirementInitEvent, requirementEventTypes } from "./requirements/events";
import { Run, RunContext, RunMiddlewareType } from "@/context";
import { Emitter } from "@/emitter";
import { FrameworkError } from "@/errors";
import { AnyTool } from "@/tools";
import { MaybeAsync } from "@/utils";
import { toSafeWord } from "@/utils/strings";

export interface Rule {
  /** A tool that the requirement apply to. */
  target: string;
  /** Can the agent use the tool? */
  allowed?: boolean;
  /** Reason for the rule. */
  reason?: string | null;
  /** Prevent the agent from terminating. */
  preventStop?: boolean;
  /** Must the agent use the tool? */
  forced?: boolean;
  /** Completely omit the tool. */
  hidden?: boolean;
}

export abstract class Requirement<T = any> {
  public readonly name: string;
  public state: Record<string, any>;
  public enabled = true;
  public middlewares: RunMiddlewareType[] = [];

  private _priority = 10;
  private _emitter?: Emitter;

  constructor(...args: any[]) {
    this.state = {};
  }

  get emitter(): Emitter {
    if (!this._emitter) {
      this._emitter = this.createEmitter();
    }
    return this._emitter;
  }

  private createEmitter(): Emitter {
    return Emitter.root().child({
      namespace: ["requirement", toSafeWord(this.name)],
      creator: this,
      events: requirementEventTypes,
    });
  }

  get priority(): number {
    return this._priority;
  }

  set priority(value: number) {
    if (value <= 0) {
      throw new Error("Priority must be a positive integer.");
    }
    this._priority = value;
  }

  abstract run(state: T): Run<Rule[]>;

  async init(options: { tools: AnyTool[]; ctx: RunContext }): Promise<void> {
    await this.emitter.emit("init", new RequirementInitEvent(options.tools));
  }

  async clone(): Promise<this> {
    const instance = Object.create(Object.getPrototypeOf(this));
    instance.name = this.name;
    instance.priority = this.priority;
    instance.enabled = this.enabled;
    instance.state = { ...this.state };
    return instance;
  }

  toJsonSafe(): Record<string, any> {
    return {
      name: this.name,
      priority: this.priority,
      enabled: this.enabled,
      state: this.state,
      className: this.constructor.name,
    };
  }
}

export function runWithContext<TSelf extends Requirement<any>, T>(
  fn: (self: TSelf, input: T, context: RunContext) => Promise<Rule[]>,
): (self: TSelf, input: T) => Run<Rule[]> {
  return function decorated(this: TSelf, input: T): Run<Rule[]> {
    const handler = async (context: RunContext): Promise<Rule[]> => {
      return await fn(this, input, context);
    };

    const runParams =
      input && typeof input === "object" && "modelDump" in input
        ? (input as any).modelDump()
        : input;

    return RunContext.enter(this, handler, runParams).middleware(...this.middlewares);
  };
}

export type RequirementFn<T> = MaybeAsync<(state: T, context: RunContext) => Rule[]>;

export function requirement<T = any>(
  options: {
    name?: string;
    targets?: MultiTargetType;
  } = {},
): (fn: RequirementFn<T>) => Requirement<T> {
  return function createRequirement(fn: RequirementFn<T>): Requirement<T> {
    const reqName = options.name || fn.name;
    const reqTargets = extractTargets(options.targets);

    class FunctionRequirement extends Requirement<T> {
      name = reqName || fn.name;

      @runWithContext
      async run(state: T, context: RunContext): Promise<Rule[]> {
        const result = fn(state, context);
        if (result instanceof Promise) {
          return await result;
        }
        return result;
      }

      async init(options: { tools: AnyTool[]; ctx: RunContext }): Promise<void> {
        await super.init(options);
        assertAllRulesFound({ targets: reqTargets, tools: options.tools });
      }
    }

    return new FunctionRequirement();
  };
}

export class RequirementError extends FrameworkError {
  private _requirement?: Requirement<any>;

  constructor(
    message = "Framework error",
    options: {
      requirement?: Requirement<any>;
      cause?: Error;
      context?: Record<string, any>;
    } = {},
  ) {
    super(message, {
      isFatal: true,
      isRetryable: false,
      cause: options.cause,
      context: options.context,
    });
    this._requirement = options.requirement;
  }

  get requirement(): Requirement<any> | undefined {
    return this._requirement;
  }
}
