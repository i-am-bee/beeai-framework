/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "@/internals/serializable.js";
import type { ReActAgentTemplates } from "@/agents/react/types.js";
import {
  ReActAgentRunIteration,
  ReActAgentCallbacks,
  ReActAgentIterationToolResult,
  ReActAgentMeta,
  ReActAgentRunInput,
  ReActAgentRunOptions,
} from "@/agents/react/types.js";
import type { ReActAgent, ReActAgentInput } from "@/agents/react/agent.js";
import { RetryCounter } from "@/internals/helpers/counter.js";
import { AgentError } from "@/agents/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { BaseMemory } from "@/memory/base.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { Cache } from "@/cache/decoratorCache.js";
import { mapObj } from "@/internals/helpers/object.js";
import { PromptTemplate } from "@/template.js";

export interface ReActAgentRunnerLLMInput {
  meta: ReActAgentMeta;
  signal: AbortSignal;
  emitter: Emitter<ReActAgentCallbacks>;
}

export interface ReActAgentRunnerToolInput {
  state: ReActAgentIterationToolResult;
  meta: ReActAgentMeta;
  signal: AbortSignal;
  emitter: Emitter<ReActAgentCallbacks>;
}

export abstract class BaseRunner extends Serializable {
  public memory!: BaseMemory;
  public readonly iterations: ReActAgentRunIteration[] = [];
  protected readonly failedAttemptsCounter: RetryCounter;

  constructor(
    protected readonly input: ReActAgentInput,
    protected readonly options: ReActAgentRunOptions,
    protected readonly run: GetRunContext<ReActAgent>,
  ) {
    super();
    this.failedAttemptsCounter = new RetryCounter(options?.execution?.totalMaxRetries, AgentError);
  }

  async createIteration() {
    const meta: ReActAgentMeta = { iteration: this.iterations.length + 1 };
    const maxIterations = this.options?.execution?.maxIterations ?? Infinity;

    if (meta.iteration > maxIterations) {
      throw new AgentError(
        `Agent was not able to resolve the task in ${maxIterations} iterations.`,
        [],
        {
          isFatal: true,
          isRetryable: false,
          context: { iterations: this.iterations.map((iteration) => iteration.state) },
        },
      );
    }

    const emitter = this.run.emitter.child({ groupId: `iteration-${meta.iteration}` });
    const iteration = await this.llm({ emitter, signal: this.run.signal, meta });
    this.iterations.push(iteration);

    return {
      emitter,
      state: iteration.state,
      meta,
      signal: this.run.signal,
    };
  }

  async init(input: ReActAgentRunInput) {
    this.memory = await this.initMemory(input);
  }

  abstract llm(input: ReActAgentRunnerLLMInput): Promise<ReActAgentRunIteration>;

  abstract tool(input: ReActAgentRunnerToolInput): Promise<{ output: string; success: boolean }>;

  public abstract get defaultTemplates(): ReActAgentTemplates;

  @Cache({ enumerable: false })
  public get templates(): ReActAgentTemplates {
    const overrides = this.input.templates ?? {};

    return mapObj(this.defaultTemplates)(
      (key, defaultTemplate: ReActAgentTemplates[typeof key]) => {
        const override = overrides[key] ?? defaultTemplate;
        if (override instanceof PromptTemplate) {
          return override;
        }
        return override(defaultTemplate) ?? defaultTemplate;
      },
    );
  }

  protected abstract initMemory(input: ReActAgentRunInput): Promise<BaseMemory>;

  createSnapshot() {
    return {
      input: shallowCopy(this.input),
      options: shallowCopy(this.options),
      memory: this.memory,
      failedAttemptsCounter: this.failedAttemptsCounter,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
