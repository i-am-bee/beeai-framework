/**
 * Copyright 2025 IBM Corp.
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

import { Serializable } from "@/internals/serializable.js";
import type { BeeAgentTemplates } from "@/agents/bee/types.js";
import {
  BeeAgentRunIteration,
  BeeCallbacks,
  BeeIterationToolResult,
  BeeMeta,
  BeeRunInput,
  BeeRunOptions,
} from "@/agents/bee/types.js";
import type { BeeAgent, BeeInput } from "@/agents/bee/agent.js";
import { RetryCounter } from "@/internals/helpers/counter.js";
import { AgentError } from "@/agents/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { BaseMemory } from "@/memory/base.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { PromptTemplate } from "@/template.js";
import { Cache } from "@/cache/decoratorCache.js";
import { mapValues } from "remeda";

export interface BeeRunnerLLMInput {
  meta: BeeMeta;
  signal: AbortSignal;
  emitter: Emitter<BeeCallbacks>;
}

export interface BeeRunnerToolInput {
  state: BeeIterationToolResult;
  meta: BeeMeta;
  signal: AbortSignal;
  emitter: Emitter<BeeCallbacks>;
}

export abstract class BaseRunner extends Serializable {
  public memory!: BaseMemory;
  public readonly iterations: BeeAgentRunIteration[] = [];
  protected readonly failedAttemptsCounter: RetryCounter;

  constructor(
    protected readonly input: BeeInput,
    protected readonly options: BeeRunOptions,
    protected readonly run: GetRunContext<BeeAgent>,
  ) {
    super();
    this.failedAttemptsCounter = new RetryCounter(options?.execution?.totalMaxRetries, AgentError);
  }

  async createIteration() {
    const meta: BeeMeta = { iteration: this.iterations.length + 1 };
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

  async init(input: BeeRunInput) {
    this.memory = await this.initMemory(input);
  }

  abstract llm(input: BeeRunnerLLMInput): Promise<BeeAgentRunIteration>;

  abstract tool(input: BeeRunnerToolInput): Promise<{ output: string; success: boolean }>;

  public abstract get defaultTemplates(): BeeAgentTemplates;

  @Cache({ enumerable: false })
  public get templates(): BeeAgentTemplates {
    const templatesUpdate = this.input.templates ?? {};

    return mapValues(this.defaultTemplates, (defaultTemplate, key) => {
      if (!templatesUpdate[key]) {
        return defaultTemplate;
      }
      if (templatesUpdate[key] instanceof PromptTemplate) {
        return templatesUpdate[key];
      }
      const update = templatesUpdate[key] as (template: typeof defaultTemplate) => typeof template;
      return update(defaultTemplate);
    }) as BeeAgentTemplates;
  }

  protected abstract initMemory(input: BeeRunInput): Promise<BaseMemory>;

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
