/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";
import { AgentMeta } from "@/agents/types.js";
import { Serializable } from "@/internals/serializable.js";
import { GetRunContext, RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { BaseMemory } from "@/memory/base.js";

export class AgentError extends FrameworkError {}

export interface BaseAgentRunOptions {
  signal?: AbortSignal;
}

export abstract class BaseAgent<
  TInput,
  TOutput,
  TOptions extends BaseAgentRunOptions = BaseAgentRunOptions,
> extends Serializable {
  protected isRunning = false;

  public abstract readonly emitter: Emitter<unknown>;

  public run(
    ...[input, options]: Partial<TOptions> extends TOptions
      ? [input: TInput, options?: TOptions]
      : [input: TInput, options: TOptions]
  ) {
    if (this.isRunning) {
      throw new AgentError("Agent is already running!");
    }

    return RunContext.enter(
      this,
      { signal: options?.signal, params: [input, options] as const },
      async (context) => {
        try {
          this.isRunning = true;
          // @ts-expect-error
          return await this._run(input, options, context);
        } catch (e) {
          if (e instanceof AgentError) {
            throw e;
          } else {
            throw new AgentError(`The Agent has encountered an error.`, [e]);
          }
        } finally {
          this.isRunning = false;
        }
      },
    );
  }

  protected abstract _run(
    input: TInput,
    options: TOptions,
    run: GetRunContext<typeof this>,
  ): Promise<TOutput>;

  destroy() {
    this.emitter.destroy();
  }

  public abstract set memory(memory: BaseMemory);
  public abstract get memory(): BaseMemory;

  public get meta(): AgentMeta {
    return {
      name: this.constructor.name ?? "BaseAgent",
      description: "",
      tools: [],
    };
  }

  createSnapshot() {
    return { isRunning: false, emitter: this.emitter };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
