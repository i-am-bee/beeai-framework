/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { Message, AssistantMessage } from "@/backend/message.js";

// TS exports the message union type as `Message` (Python calls the
// equivalent `AnyMessage`). Alias it locally to keep this file's naming
// consistent with the Python reference implementation.
type AnyMessage = Message;
import { GetRunContext, MiddlewareType, Run, RunContext, RunInstance } from "@/context.js";

/**
 * Options accepted by a Runnable's `run` method.
 *
 * Mirrors Python's `RunnableOptions` (TypedDict).
 */
export interface RunnableOptions {
  /** Abort signal for cancelling the run. */
  signal?: AbortSignal;
  /** Additional context to pass into the runnable. */
  context?: Record<string, any>;
}

/**
 * Output produced by a Runnable.
 *
 * Mirrors Python's `RunnableOutput` (pydantic BaseModel with `extra="allow"`).
 */
export class RunnableOutput {
  /** The runnable output messages. */
  public output: AnyMessage[];

  /** Additional context returned by the runnable. */
  public context: Record<string, any>;

  // Allow arbitrary extra fields, mirroring Python's `extra="allow"`.
  [key: string]: any;

  constructor(input: { output: AnyMessage[]; context?: Record<string, any> } & Record<string, any>) {
    const { output, context, ...extra } = input;
    this.output = output;
    this.context = context ?? {};
    Object.assign(this, extra);
  }

  /**
   * Returns the latest message in output, with a fallback if it is not defined.
   */
  get lastMessage(): AnyMessage {
    const last = this.output.at(-1);
    return last ?? new AssistantMessage("");
  }
}

/**
 * A unit of work that can be invoked using a stable interface.
 *
 * Mirrors Python's `Runnable[R]` ABC. Concrete subclasses (agents, workflows,
 * chat models, ...) implement `run` using `RunContext.enter`, matching the
 * pattern already used elsewhere in this codebase (see `@/context.js`).
 */
export abstract class Runnable<R extends RunnableOutput = RunnableOutput> implements RunInstance {
  protected readonly _middlewares: MiddlewareType<this>[];

  constructor(middlewares: MiddlewareType<any>[] = []) {
    this._middlewares = middlewares as MiddlewareType<this>[];
  }

  /**
   * Execute the runnable.
   *
   * @param input The input messages to the runnable.
   * @param options Execution options (signal, context).
   */
  abstract run(input: AnyMessage[], options?: RunnableOptions): Run<R, this>;

  /** The event emitter for the runnable. */
  abstract get emitter(): Emitter<any>;

  /** The list of middleware to be used when executing the runnable. */
  get middlewares(): MiddlewareType<this>[] {
    return this._middlewares;
  }
}

export type AnyRunnable = Runnable<any>;

/**
 * Helper to wrap a runnable's handler logic into a `RunContext`-managed execution.
 *
 * This is the TypeScript equivalent of Python's `runnable_entry` decorator.
 * Python relies on a decorator to auto-wrap `run()` with `RunContext.enter()`;
 * TS decorators are still experimental/inconsistent across build tools, so
 * this is implemented as a plain helper function instead, called directly
 * from within a concrete `run()` implementation:
 *
 * ```ts
 * class MyRunnable extends Runnable<RunnableOutput> {
 *   run(input: AnyMessage[], options?: RunnableOptions) {
 *     return runnableEntry(this, input, options, async (ctx, input) => {
 *       // `input` here reflects any middleware modifications to ctx.runParams.
 *       // ... runnable logic using ctx / input ...
 *       return new RunnableOutput({ output: [...] });
 *     });
 *   }
 *
 *   get emitter() {
 *     return this._emitter;
 *   }
 * }
 * ```
 */
export function runnableEntry<I extends Runnable<R>, R extends RunnableOutput, M extends AnyMessage = AnyMessage>(
  instance: I,
  input: M[],
  options: RunnableOptions | undefined,
  handler: (context: GetRunContext<I>, input: M[]) => Promise<R>,
): Run<R, I> {
  return RunContext.enter(
    instance,
    { params: input, signal: options?.signal },
    // Read the (possibly middleware-modified) input from `context.runParams`
    // rather than closing over the original `input` argument, so that
    // middleware mutating runParams before execution actually takes effect.
    // This matches Python's `runnable_entry`, which reads
    // `ctx.run_params.get("input", ...)` for the same reason.
    (context) => handler(context, context.runParams ?? input),
  )
    .middleware(...instance.middlewares)
    .context(options?.context ?? {});
}