/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Middleware, RunContext, RunContextCallbacks, RunInstance } from "@/context.js";
import { Callback, Emitter, EventMeta } from "@/emitter/emitter.js";
import { FrameworkError } from "@/errors.js";
import { Logger } from "@/logger/logger.js";
import { BaseAgent } from "@/agents/base.js";
import type { AnyConstructable } from "@/internals/types.js";
import { capitalize } from "remeda";
import { ChatModel } from "@/backend/chat.js";
import { Tool } from "@/tools/base.js";
import { Requirement } from "@/agents/requirement/requirements/requirement.js";
import type { InferCallbackValue } from "@/emitter/types.js";
import { Serializer } from "@/serializer/serializer.js";
import { isPrimitive } from "@/internals/helpers/guards.js";

/**
 * Information about how deep the given entity is in the execution tree.
 */
export interface TraceLevel {
  /** Relative depth to the included (observed) elements */
  relative: number;
  /** Absolute depth from the root */
  absolute: number;
}

/**
 * Input for custom formatter function
 */
export interface GlobalTrajectoryMiddlewareFormatterInput {
  prefix: string;
  className: string;
  eventName: string;
  instanceName: string | null;
}

export interface GlobalTrajectoryMiddlewareCallbacks {
  start: Callback<{
    message: string;
    level: TraceLevel;
    origin: [InferCallbackValue<RunContextCallbacks["start"]>, EventMeta];
  }>;
  success: Callback<{
    message: string;
    level: TraceLevel;
    origin: [InferCallbackValue<RunContextCallbacks["success"]>, EventMeta];
  }>;
  error: Callback<{
    message: string;
    level: TraceLevel;
    origin: [InferCallbackValue<RunContextCallbacks["error"]>, EventMeta];
  }>;
  finish: Callback<{
    message: string;
    level: TraceLevel;
    origin: [InferCallbackValue<RunContextCallbacks["finish"]>, EventMeta];
  }>;
}

type OutputTargetFn = (message: string) => void;
type OutputTarget = Logger | OutputTargetFn;

export interface GlobalTrajectoryMiddlewareOptions {
  /** Specify output target: 'console', Logger instance, custom function, or null to disable */
  target?: OutputTarget;
  /** List of classes to include in the trajectory */
  included?: AnyConstructable[];
  /** List of classes to exclude from the trajectory */
  excluded?: AnyConstructable[];
  /** Use pretty formatting for the trajectory */
  pretty?: boolean;
  /** Customize how instances of individual classes should be printed */
  prefixByType?: Map<AnyConstructable, string>;
  /** Enable/Disable the logging */
  enabled?: boolean;
  /** Whether to observe trajectories of nested run contexts */
  matchNested?: boolean;
  /** Defines a priority for registered events */
  emitterPriority?: number;
  /** Custom formatter function */
  formatter?: (input: GlobalTrajectoryMiddlewareFormatterInput) => string;
}

/**
 * Middleware for capturing and logging execution flow of agents, tools, and models.
 * Provides hierarchical visualization with indentation to show the call stack.
 */
export class GlobalTrajectoryMiddleware<T extends RunInstance = any> extends Middleware<T> {
  protected enabled: boolean;
  protected included: AnyConstructable[];
  protected excluded: AnyConstructable[];
  protected cleanups: (() => void)[] = [];
  protected target: (message: string) => void;
  protected ctx: RunContext<T> | null = null;
  protected pretty: boolean;
  protected traceLevel = new Map<string, TraceLevel>();
  protected prefixByType: Map<any, string>;
  protected matchNested: boolean;
  protected emitterPriority: number;
  protected formatter: (input: GlobalTrajectoryMiddlewareFormatterInput) => string;
  public readonly emitter: Emitter<GlobalTrajectoryMiddlewareCallbacks>;

  constructor(options: GlobalTrajectoryMiddlewareOptions = {}) {
    super();

    this.enabled = options.enabled ?? true;
    this.included = options.included ?? [];
    this.excluded = options.excluded ?? [];
    this.target = this.createTarget(options.target);
    this.pretty = options.pretty ?? false;
    this.matchNested = options.matchNested ?? true;
    this.emitterPriority = options.emitterPriority ?? -1; // run later

    // Default prefixes
    this.prefixByType = new Map([
      [BaseAgent, "🤖 "],
      [ChatModel, "💬 "],
      [Tool, "🛠️ "],
      [Requirement, "🔎 "],
      ...((options.prefixByType ? Array.from(options.prefixByType.entries()) : []) as any),
    ]);

    this.formatter =
      options.formatter ??
      ((x) => `${x.prefix}${x.className}[${x.instanceName || x.className}][${x.eventName}]`);

    this.emitter = Emitter.root.child<GlobalTrajectoryMiddlewareCallbacks>({
      namespace: ["middleware", "globalTrajectory"],
    });
  }

  /**
   * Bind the middleware to a run context
   */
  bind(ctx: RunContext<T>): void {
    // Cleanup previous bindings
    while (this.cleanups.length > 0) {
      this.cleanups.pop()!();
    }

    this.traceLevel.clear();
    this.traceLevel.set(ctx.runId, { relative: 0, absolute: 0 });
    this.ctx = ctx;

    this.bindEmitter(ctx.emitter);
  }

  private createTarget(input?: OutputTarget): OutputTargetFn {
    if (input === null || input === undefined) {
      // eslint-disable-next-line no-console
      return (msg) => console.log(msg);
    } else if (input instanceof Logger) {
      return (msg) => input.debug(msg);
    } else {
      return input;
    }
  }

  private bindEmitter(emitter: Emitter<any>): void {
    // Track all events for trace ID logging
    this.cleanups.push(
      emitter.match("*.*", (_, event) => this.logTraceId(event), {
        matchNested: true,
      }),
    );

    // Handle top-level events
    const handleTopLevelEvent = async (data: any, meta: EventMeta) => {
      if (!(meta.creator instanceof RunContext)) {
        return;
      }
      if (!meta.trace) {
        return;
      }

      this.logTraceId(meta);
      if (!this.isAllowed(meta)) {
        return;
      }
      if (!this.enabled) {
        return;
      }

      const eventName = capitalize(meta.name) as "Start" | "Success" | "Error" | "Finish";
      await this[`onInternal${eventName}` as const].call(this, data, meta);
    };

    this.cleanups.push(
      emitter.match(
        (event) => ["start", "success", "error", "finish"].includes(event.name),
        handleTopLevelEvent,
        {
          matchNested: false,
          isBlocking: true,
          priority: this.emitterPriority,
        },
      ),
    );

    // Handle nested events if enabled
    if (this.matchNested) {
      const handleNestedEvent = async (data: any, meta: EventMeta) => {
        if (!(meta.creator instanceof RunContext)) {
          return;
        }
        if (meta.creator.emitter !== emitter) {
          await handleTopLevelEvent(data, meta);
          this.bindEmitter(meta.creator.emitter);
        }
      };

      this.cleanups.push(
        emitter.match((event) => event.name === "start", handleNestedEvent, {
          matchNested: true,
          isBlocking: true,
          priority: this.emitterPriority,
        }),
      );
    }
  }

  private logTraceId(meta: EventMeta): void {
    if (!meta.trace?.runId) {
      return;
    }
    if (this.traceLevel.has(meta.trace.runId)) {
      return;
    }
    if (meta.trace.parentRunId === meta.trace.runId) {
      return;
    }

    if (meta.trace.parentRunId) {
      const allowed = this.isAllowed(meta);
      const parentTrace = this.traceLevel.get(meta.trace.parentRunId) ?? {
        relative: 0,
        absolute: 0,
      };
      this.traceLevel.set(meta.trace.runId, {
        relative: parentTrace.relative + (allowed ? 1 : 0),
        absolute: parentTrace.absolute + 1,
      });
    }
  }

  private isAllowed(meta: EventMeta): boolean {
    let target: any = meta.creator;
    if (target instanceof RunContext) {
      target = target.instance;
    }

    for (const excluded of this.excluded) {
      if (target instanceof excluded) {
        return false;
      }
    }

    if (this.included.length === 0) {
      return true;
    }

    return this.included.some((included) => target instanceof included);
  }

  private extractName(meta: EventMeta): string {
    let target: any = meta.creator;
    if (target instanceof RunContext) {
      target = target.instance;
    }

    const className = target.constructor.name;
    let targetName: string | null = null;

    if (target instanceof BaseAgent && target.meta?.name) {
      targetName = target.meta.name;
    } else if ("name" in target && typeof target.name === "string") {
      targetName = target.name;
    }

    let prefix = "";
    for (const [type, typePrefix] of this.prefixByType.entries()) {
      if (target instanceof type) {
        prefix = typePrefix;
        break;
      }
    }

    const input: GlobalTrajectoryMiddlewareFormatterInput = {
      prefix,
      className,
      instanceName: targetName,
      eventName: meta.name,
    };

    return this.formatter(input);
  }

  private formatPrefix(meta: EventMeta): string {
    if (!meta.trace) {
      return "";
    }

    const indent = this.getTraceLevel(meta, "self").relative;
    const indentParent = this.getTraceLevel(meta, "parent").relative;
    const indentDiff = indent - indentParent;

    let prefix = "";
    prefix += "  ".repeat(indentParent * 2);

    if (meta.name !== "start" && indent) {
      prefix += "<";
    }

    prefix += "--".repeat(indentDiff);

    if (meta.name === "start" && prefix && indent) {
      prefix += ">";
    }

    if (prefix) {
      prefix = `${prefix} `;
    }

    const name = this.extractName(meta);
    return `${prefix}${name}: `;
  }

  private getTraceLevel(meta: EventMeta, type: "self" | "parent" = "self"): TraceLevel {
    if (!meta.trace) {
      return { relative: 0, absolute: 0 };
    }

    const runId = type === "parent" ? meta.trace.parentRunId || "" : meta.trace.runId;
    return this.traceLevel.get(runId) ?? { relative: 0, absolute: 0 };
  }

  private async formatPayload(value: any): Promise<string> {
    if (isPrimitive(value)) {
      return String(value);
    }

    if (value instanceof FrameworkError) {
      return value.explain();
    }

    const serialized = await Serializer.serialize(value);
    return JSON.stringify(
      await Serializer.deserialize(serialized, [], true),
      (() => {
        const excludedKeys = new Set(["emitter", "cleanups", "creator", "listeners"]);
        return (key, value) => {
          if (excludedKeys.has(key)) {
            return undefined;
          }
          if (value && value instanceof Tool) {
            return value.name;
          }
          return value;
        };
      })(),
      this.pretty ? 2 : undefined,
    );
  }

  private async onInternalStart(
    payload: InferCallbackValue<RunContextCallbacks["start"]>,
    meta: EventMeta,
  ): Promise<void> {
    const prefix = this.formatPrefix(meta);
    const message = `${prefix}${await this.formatPayload(payload.input)}`;

    await this.emitter.emit("start", {
      message,
      level: this.getTraceLevel(meta),
      origin: [payload, meta],
    });

    this.target(message);
  }

  private async onInternalSuccess(
    payload: InferCallbackValue<RunContextCallbacks["success"]>,
    meta: EventMeta,
  ): Promise<void> {
    const prefix = this.formatPrefix(meta);
    // TODO: change to payload.output once available
    const message = `${prefix}${await this.formatPayload(payload)}`;

    await this.emitter.emit("success", {
      message,
      level: this.getTraceLevel(meta),
      origin: [payload, meta],
    });

    this.target(message);
  }

  private async onInternalError(
    payload: InferCallbackValue<RunContextCallbacks["error"]>,
    meta: EventMeta,
  ): Promise<void> {
    const prefix = this.formatPrefix(meta);
    const message = `${prefix}${await this.formatPayload(payload)}`;

    await this.emitter.emit("error", {
      message,
      level: this.getTraceLevel(meta),
      origin: [payload, meta],
    });

    this.target(message);
  }

  private async onInternalFinish(
    payload: InferCallbackValue<RunContextCallbacks["finish"]>,
    meta: EventMeta,
  ): Promise<void> {
    const prefix = this.formatPrefix(meta);
    const message = `${prefix}${await this.formatPayload(payload.error || payload.output)}`;
    await this.emitter.emit("finish", {
      message,
      level: this.getTraceLevel(meta),
      origin: [payload, meta],
    });
  }
}
