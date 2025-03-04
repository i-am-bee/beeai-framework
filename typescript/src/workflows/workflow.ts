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

import { z, ZodSchema } from "zod";
import { Serializable } from "@/internals/serializable.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { RunContext } from "@/context.js";
import { omit, pick, toCamelCase } from "remeda";
import { deepCopy, shallowCopy } from "@/serializer/utils.js";
import { FrameworkError, ValueError } from "@/errors.js";

// eslint-disable-next-line @typescript-eslint/no-invalid-void-type
type WorkflowStepHandlerReturn<K extends string> = K | ReservedStep | void;

export interface WorkflowRun<T extends ZodSchema, T2 extends ZodSchema, K extends string> {
  result: z.output<T2>;
  steps: WorkflowStepRes<T, K>[];
  state: z.output<T>;
}

export interface WorkflowRunOptions<K extends string> {
  start?: K;
  signal?: AbortSignal;
}

export interface WorkflowStepDef<T extends ZodSchema, K extends string> {
  schema: T;
  handler: WorkflowStepHandler<T, K>;
}

export interface WorkflowStepRes<T extends ZodSchema, K extends string> {
  name: K;
  next: K | ReservedStep;
  state: z.output<T>;
}

export interface WorkflowRunContext<T extends ZodSchema, K extends string> {
  steps: WorkflowStepRes<T, K>[];
  signal: AbortSignal;
  abort: (reason?: Error) => void;
}

export type WorkflowStepHandler<T extends ZodSchema, K extends string> = (
  state: z.output<T>,
  context: WorkflowRunContext<T, K>,
) => Promise<WorkflowStepHandlerReturn<K>> | WorkflowStepHandlerReturn<K>;

export interface WorkflowEvents<T extends ZodSchema, T2 extends ZodSchema, K extends string> {
  start: Callback<{ step: K; run: WorkflowRun<T, T2, K> }>;
  error: Callback<{
    step: K;
    error: Error;
    run: WorkflowRun<T, T2, K>;
  }>;
  success: Callback<{
    step: K;
    next: K | ReservedStep;
    state: z.output<T>;
    run: WorkflowRun<T, T2, K>;
  }>;
}

interface FlowInput<TS extends ZodSchema, TS2 extends ZodSchema = TS> {
  name?: string;
  schema: TS;
  outputSchema?: TS2;
}

type RunStep<K extends string> = K | typeof Workflow.END;
type ReservedStep =
  | typeof Workflow.START
  | typeof Workflow.NEXT
  | typeof Workflow.PREV
  | typeof Workflow.END
  | typeof Workflow.SELF;
export type FlowStepName<K extends string> = K | ReservedStep;

export class WorkflowError<
  T extends ZodSchema,
  T2 extends ZodSchema,
  K extends string,
> extends FrameworkError {
  constructor(message: string, extra?: { run?: WorkflowRun<T, T2, K>; errors?: Error[] }) {
    super(message, extra?.errors, {
      context: extra?.run ?? {},
      isRetryable: false,
      isFatal: true,
    });
  }
}

export class Workflow<
  TInput extends ZodSchema,
  TOutput extends ZodSchema = TInput,
  TKeys extends string = string,
> extends Serializable {
  public static readonly START = "__start__";
  public static readonly SELF = "__self__";
  public static readonly PREV = "__prev__";
  public static readonly NEXT = "__next__";
  public static readonly END = "__end__";

  public readonly emitter: Emitter<WorkflowEvents<TInput, TOutput, TKeys>>;

  protected readonly steps = new Map<string, WorkflowStepDef<TInput, TKeys>>();
  protected startStep: TKeys | undefined = undefined;

  constructor(protected readonly input: FlowInput<TInput, TOutput>) {
    super();
    this.emitter = Emitter.root.child({
      namespace: ["workflow", toCamelCase(input?.name ?? "")].filter(Boolean),
      creator: this,
    });
  }

  getSteps() {
    return Array.from(this.steps.keys()) as TKeys[];
  }

  get name() {
    return this.input.name ?? "";
  }

  get schemas() {
    return pick(this.input, ["schema", "outputSchema"]);
  }

  addStep<L extends string>(
    name: L,
    handler: WorkflowStepHandler<TInput | TOutput, TKeys>,
  ): Workflow<TInput, TOutput, L | TKeys>;
  addStep<L extends string, TFlow extends AnyWorkflowFlow>(
    name: L,
    flow: Workflow.pipeable<this, TFlow>,
  ): Workflow<TInput, TOutput, L | TKeys>;
  addStep<L extends string>(
    name: L,
    step: WorkflowStepHandler<TInput | TOutput, TKeys> | AnyWorkflowFlow,
  ): Workflow<TInput, TOutput, L | TKeys> {
    return this._addStep(name, step);
  }

  addStrictStep<L extends string, TI2 extends ZodSchema>(
    name: L,
    schema: TI2,
    handler: WorkflowStepHandler<TI2, TKeys>,
  ): Workflow<TInput, TOutput, L | TKeys>;
  addStrictStep<L extends string, TI2 extends ZodSchema, TFlow extends AnyWorkflowFlow>(
    name: L,
    schema: TI2,
    flow: Workflow.pipeable<Workflow<TI2, TOutput, TKeys>, TFlow>,
  ): Workflow<TInput, TOutput, L | TKeys>;
  addStrictStep<L extends string, TI2 extends ZodSchema>(
    name: L,
    schema: TI2,
    step: WorkflowStepHandler<TI2, TKeys> | AnyWorkflowFlow,
  ): Workflow<TInput, TOutput, L | TKeys> {
    return this._addStep(name, schema, step);
  }

  protected _addStep<TI2 extends ZodSchema = TInput, L extends string = TKeys>(
    name: L,
    schemaOrStep: TI2 | WorkflowStepHandler<TInput, TKeys> | AnyWorkflowFlow,
    stepOrEmpty?: WorkflowStepHandler<TI2, TKeys> | AnyWorkflowFlow,
  ): Workflow<TInput, TOutput, L | TKeys> {
    if (!name.trim()) {
      throw new ValueError(`Step name cannot be empty!`);
    }
    if (this.steps.has(name)) {
      throw new ValueError(`Step '${name}' already exists!`);
    }
    if (name === Workflow.END) {
      throw new ValueError(`The name '${name}' cannot be used!`);
    }

    const schema = (schemaOrStep && stepOrEmpty ? schemaOrStep : this.input.schema) as TInput;
    const stepOrFlow = stepOrEmpty || schemaOrStep;

    this.steps.set(name, {
      schema,
      handler: stepOrFlow instanceof Workflow ? stepOrFlow.asStep({}) : stepOrFlow,
    } as WorkflowStepDef<TInput, TKeys>);

    return this as unknown as Workflow<TInput, TOutput, L | TKeys>;
  }

  setStart(name: TKeys) {
    this.startStep = name;
    return this;
  }

  run(state: z.input<TInput>, options: WorkflowRunOptions<TKeys> = {}) {
    return RunContext.enter(
      this,
      { signal: options?.signal, params: [state, options] as const },
      async (runContext): Promise<WorkflowRun<TInput, TOutput, TKeys>> => {
        const run: WorkflowRun<TInput, TOutput, TKeys> = {
          steps: [],
          state: this.input.schema.parse(state),
          result: undefined as z.output<TOutput>,
        };
        const handlers: WorkflowRunContext<TInput, TKeys> = {
          steps: run.steps,
          signal: runContext.signal,
          abort: (reason) => runContext.abort(reason),
        };

        let next: RunStep<TKeys> =
          this.findStep(options?.start || this.startStep || this.getSteps()[0]).current ??
          Workflow.END;

        while (next && next !== Workflow.END) {
          const name = next;
          const step = this.steps.get(next);
          if (!step) {
            throw new WorkflowError(`Step '${next}' was not found.`, { run });
          }
          await runContext.emitter.emit("start", { run, step: next });
          try {
            const state = await step.schema
              .parseAsync(run.state)
              .catch(async (err: Error) => {
                throw new WorkflowError(
                  `Step '${next}' cannot be executed because the provided input doesn't adhere to the step's schema.`,
                  { run: shallowCopy(run), errors: [err] },
                );
              })
              .then(deepCopy);

            const nextStepRaw = await step.handler(state, handlers);
            if (nextStepRaw === Workflow.START) {
              next = run.steps.at(0)?.name!;
            } else if (nextStepRaw === Workflow.PREV) {
              next = this.findStep(next).prev;
            } else if (nextStepRaw === Workflow.SELF) {
              next = this.findStep(next).current;
            } else if (!nextStepRaw || nextStepRaw === Workflow.NEXT) {
              next = this.findStep(next).next || Workflow.END;
            } else {
              next = nextStepRaw;
            }

            const stepResult: WorkflowStepRes<TInput, TKeys> = {
              name,
              state,
              next,
            };
            run.steps.push(stepResult);
            run.state = stepResult.state;

            await runContext.emitter.emit("success", {
              run: shallowCopy(run),
              state: stepResult.state,
              step: stepResult.name,
              next: stepResult.next,
            });
          } catch (error) {
            await runContext.emitter.emit("error", {
              run: shallowCopy(run),
              step: next as TKeys,
              error,
            });
            throw error;
          }
        }

        run.result = await (this.input.outputSchema ?? this.input.schema)
          .parseAsync(run.state)
          .catch((err) => {
            throw new WorkflowError(
              `Workflow has ended but it's state does not adhere to the workflow's output schema.`,
              { run: shallowCopy(run), errors: [err] },
            );
          });
        return run;
      },
    );
  }

  delStep<L extends TKeys>(name: L): Workflow<TInput, TOutput, Exclude<TKeys, L>> {
    if (this.startStep === name) {
      this.startStep = undefined;
    }
    this.steps.delete(name);
    return this as unknown as Workflow<TInput, TOutput, Exclude<TKeys, L>>;
  }

  asStep<
    TInput2 extends ZodSchema = TInput,
    TOutput2 extends ZodSchema = TOutput,
    TKeys2 extends string = TKeys,
  >(overrides: {
    input?: (input: z.output<TInput2>) => z.output<TInput> | z.input<TInput>;
    output?: (output: z.output<TOutput>) => z.output<TOutput2> | z.input<TOutput2>;
    start?: TKeys;
    next?: FlowStepName<TKeys2>;
  }): WorkflowStepHandler<TInput2, TKeys | TKeys2> {
    return async (state, ctx) => {
      const mappedInput = overrides?.input ? overrides.input(state) : state;
      const result = await this.run(mappedInput, { start: overrides?.start, signal: ctx.signal });
      const mappedOutput = overrides?.output ? overrides.output(result.state) : result.state;
      Object.assign(state, mappedOutput);
      return overrides.next;
    };
  }

  protected findStep(current: TKeys) {
    const steps = this.getSteps();
    const index = steps.indexOf(current);
    return {
      prev: steps[index - 1],
      current: steps[index],
      next: steps[index + 1],
    };
  }

  createSnapshot() {
    return {
      input: omit(this.input, ["schema", "outputSchema"]),
      emitter: this.emitter,
      startStep: this.startStep,
      steps: this.steps,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
    this.input.schema ??= z.any() as unknown as TInput;
    this.input.outputSchema ??= z.any() as unknown as TOutput;
  }
}

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace Workflow {
  export type pipeable<T extends AnyWorkflowFlow, T2 extends AnyWorkflowFlow> =
    Workflow.state<T> extends Workflow.input<T2> ? T2 : never;
  export type input<T> = T extends Workflow<infer A, any, any> ? z.input<A> : never;
  export type state<T> = T extends Workflow<infer A, any, any> ? z.output<A> : never;
  export type output<T> = T extends Workflow<any, infer A, any> ? z.output<A> : never;
  export type run<T> = T extends Workflow<infer A, infer B, infer C> ? WorkflowRun<A, B, C> : never;
}

export type AnyWorkflowFlow = Workflow<any, any, any>;
