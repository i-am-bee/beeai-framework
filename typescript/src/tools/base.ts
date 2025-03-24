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

import { FrameworkError } from "@/errors.js";
import * as R from "remeda";
import { Retryable, RetryableConfig } from "@/internals/helpers/retryable.js";
import { Serializable } from "@/internals/serializable.js";
import { Task } from "promise-based-task";
import { Cache, ObjectHashKeyFn, WeakRefKeyFn } from "@/cache/decoratorCache.js";
import { BaseCache } from "@/cache/base.js";
import { NullCache } from "@/cache/nullCache.js";
import type { ErrorObject, ValidateFunction } from "ajv";
import {
  AnyToolSchemaLike,
  createSchemaValidator,
  FromSchemaLike,
  FromSchemaLikeRaw,
  toJsonSchema,
  validateSchema,
} from "@/internals/helpers/schema.js";
import { validate } from "@/internals/helpers/general.js";
import { z, ZodSchema } from "zod";
import { Emitter } from "@/emitter/emitter.js";
import { Callback } from "@/emitter/types.js";
import { GetRunContext, RunContext } from "@/context.js";
import { shallowCopy } from "@/serializer/utils.js";
import { toCamelCase } from "remeda";

export class ToolError extends FrameworkError {}

export class ToolInputValidationError extends ToolError {
  validationErrors: ErrorObject[];

  constructor(message: string, validationErrors: ErrorObject[] = []) {
    super(message, []);
    this.validationErrors = validationErrors;
  }
}

export interface RetryOptions {
  maxRetries?: number;
  factor?: number;
}

export interface BaseToolOptions<TOutput = any> {
  retryOptions?: RetryOptions;
  fatalErrors?: ErrorConstructor[];
  cache?: BaseCache<Task<TOutput>> | false;
}

export interface BaseToolRunOptions {
  retryOptions?: RetryOptions;
  signal?: AbortSignal;
}

export abstract class ToolOutput extends Serializable {
  abstract getTextContent(): string;
  abstract isEmpty(): boolean;

  toString() {
    return this.getTextContent();
  }
}

export class StringToolOutput extends ToolOutput {
  constructor(
    public readonly result = "",
    public readonly ctx?: Record<string, any>,
  ) {
    super();
    this.result = result ?? "";
  }

  static {
    this.register();
  }

  isEmpty() {
    return !this.result;
  }

  @Cache({
    cacheKey: WeakRefKeyFn.from<StringToolOutput>((self) => [self.result]),
  })
  getTextContent(): string {
    return this.result.toString();
  }

  createSnapshot() {
    return {
      result: this.result,
      ctx: this.ctx,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

export class JSONToolOutput<T> extends ToolOutput {
  constructor(
    public readonly result: T,
    public readonly ctx?: Record<string, any>,
  ) {
    super();
  }

  static {
    this.register();
  }

  isEmpty() {
    return !this.result || R.isEmpty(this.result);
  }

  getTextContent(): string {
    return JSON.stringify(this.result);
  }

  createSnapshot() {
    return {
      result: this.result,
      ctx: this.ctx,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

export interface ToolSnapshot<TOutput extends ToolOutput, TOptions extends BaseToolOptions> {
  name: string;
  description: string;
  options: TOptions;
  cache: BaseCache<Task<TOutput>>;
  emitter: Emitter<any>;
}

export type InferToolOutput<T extends AnyTool> = T extends Tool<infer A, any, any> ? A : never;
export type ToolInput<T extends AnyTool> = FromSchemaLike<Awaited<ReturnType<T["inputSchema"]>>>;
export type ToolInputRaw<T extends AnyTool> = FromSchemaLikeRaw<
  Awaited<ReturnType<T["inputSchema"]>>
>;

type ToolConstructorParameters<TOptions extends BaseToolOptions> =
  Partial<TOptions> extends TOptions ? [options?: TOptions] : [options: TOptions];

export interface ToolEvents<
  TInput extends Record<string, any> = Record<string, any>,
  TOutput extends ToolOutput = ToolOutput,
> {
  start: Callback<{ input: TInput; options: unknown }>;
  success: Callback<{ output: TOutput; input: TInput; options: unknown }>;
  error: Callback<{ input: TInput; error: ToolError | ToolInputValidationError; options: unknown }>;
  retry: Callback<{ error: ToolError | ToolInputValidationError; input: TInput; options: unknown }>;
  finish: Callback<null>;
}

export type ToolEmitter<
  A extends Record<string, any>,
  B extends ToolOutput,
  C = Record<never, never>,
> = Emitter<ToolEvents<A, B> & Omit<C, keyof ToolEvents>>;

export abstract class Tool<
  TOutput extends ToolOutput = ToolOutput,
  TOptions extends BaseToolOptions = BaseToolOptions,
  TRunOptions extends BaseToolRunOptions = BaseToolRunOptions,
> extends Serializable {
  abstract name: string;
  abstract description: string;

  public readonly cache: BaseCache<Task<TOutput>>;
  public readonly options: TOptions;

  public static contextKeys = {
    Memory: Symbol("Memory"),
  } as const;

  public abstract readonly emitter: Emitter<ToolEvents<any, TOutput>>;

  abstract inputSchema(): Promise<AnyToolSchemaLike> | AnyToolSchemaLike;

  constructor(...args: ToolConstructorParameters<TOptions>) {
    super();

    const [options] = args;
    this.options = options ?? ({} as TOptions);
    this.cache = options?.cache ? options.cache : new NullCache();
  }

  protected toError(e: Error, context: any) {
    if (e instanceof ToolError) {
      Object.assign(e.context, context);
      return e;
    } else {
      return new ToolError(`Tool "${this.name}" has occurred an error!`, [e], {
        context,
      });
    }
  }

  run(input: ToolInputRaw<this>, options: Partial<TRunOptions> = {}) {
    input = shallowCopy(input);
    options = shallowCopy(options);

    return RunContext.enter(
      this,
      { signal: options?.signal, params: [input, options] as const },
      async (run) => {
        const meta = { input, options };
        let errorPropagated = false;

        try {
          input = Object.assign({ ref: input }, { ref: await this.parse(input) }).ref;

          const output = await new Retryable({
            executor: async () => {
              errorPropagated = false;
              await run.emitter.emit("start", { ...meta });
              return this.cache.enabled
                ? await this._runCached(input, options, run)
                : await this._run(input, options, run);
            },
            onError: async (error) => {
              errorPropagated = true;
              await run.emitter.emit("error", {
                error: this.toError(error, meta),
                ...meta,
              });
              if (this.options.fatalErrors?.some((cls) => error instanceof cls)) {
                throw error;
              }
            },
            onRetry: async (_, error) => {
              await run.emitter.emit("retry", { ...meta, error: this.toError(error, meta) });
            },
            config: {
              ...this._createRetryOptions(options?.retryOptions),
              signal: options?.signal,
            },
          }).get();

          await run.emitter.emit("success", { output, ...meta });
          return output;
        } catch (e) {
          const error = this.toError(e, meta);
          if (!errorPropagated) {
            await run.emitter.emit("error", {
              error,
              options,
              input,
            });
          }
          throw error;
        } finally {
          await run.emitter.emit("finish", null);
        }
      },
    );
  }

  protected async _runCached(
    input: ToolInput<this>,
    options: Partial<TRunOptions>,
    run: GetRunContext<this>,
  ): Promise<TOutput> {
    const key = ObjectHashKeyFn({
      input,
      options: R.omit(options ?? ({} as TRunOptions), ["signal", "retryOptions"]),
    });

    const cacheEntry = await this.cache.get(key);
    if (cacheEntry !== undefined) {
      return cacheEntry!;
    }

    const task = new Task<TOutput, Error>();
    await this.cache.set(key, task);
    this._run(input, options, run)
      .then((req) => task.resolve(req))
      .catch(async (err) => {
        void task.reject(err);
        await this.cache.delete(key);
      });
    return task;
  }

  public async clearCache() {
    await this.cache.clear();
  }

  protected abstract _run(
    arg: ToolInput<this>,
    options: Partial<TRunOptions>,
    run: GetRunContext<typeof this>,
  ): Promise<TOutput>;

  async getInputJsonSchema() {
    return toJsonSchema(await this.inputSchema());
  }

  static isTool(value: unknown): value is Tool {
    return value instanceof Tool && "name" in value && "description" in value;
  }

  private _createRetryOptions(...overrides: (RetryOptions | undefined)[]): RetryableConfig {
    const defaultOptions: Required<RetryOptions> = {
      maxRetries: 0,
      factor: 1,
    };

    return R.pipe(
      [defaultOptions, this.options.retryOptions, ...overrides],
      R.filter(R.isTruthy),
      R.map((input: RetryOptions) => {
        const options: RetryableConfig = {
          maxRetries: input.maxRetries ?? defaultOptions.maxRetries,
          factor: input.factor ?? defaultOptions.maxRetries,
        };
        return R.pickBy(options, R.isDefined);
      }),
      R.mergeAll,
    ) as RetryableConfig;
  }

  async parse(input: unknown | ToolInputRaw<this> | ToolInput<this>): Promise<ToolInput<this>> {
    const schema = await this.inputSchema();
    if (schema) {
      validateSchema(schema, {
        context: {
          tool: this.constructor.name,
          hint: `To do post-validation override the '${this.validateInput.name}' method.`,
          schema,
          isFatal: true,
          isRetryable: false,
        },
      });
    }

    const copy = shallowCopy(input);
    this.preprocessInput(copy);
    this.validateInput(schema, copy);
    return copy;
  }

  // eslint-disable-next-line unused-imports/no-unused-vars
  protected preprocessInput(rawInput: unknown): void {}

  protected validateInput(
    schema: AnyToolSchemaLike,
    rawInput: unknown,
  ): asserts rawInput is ToolInput<this> {
    const validator = createSchemaValidator(schema) as ValidateFunction<ToolInput<this>>;
    const success = validator(rawInput);
    if (!success) {
      throw new ToolInputValidationError(
        [
          `The received tool input does not match the expected schema.`,
          `Input Schema: "${JSON.stringify(toJsonSchema(schema))}"`,
          `Validation Errors: ${JSON.stringify(validator.errors)}`,
        ].join("\n"),
        // ts doesn't infer that when success is false `validator.errors` is defined
        validator.errors!,
      );
    }
  }

  createSnapshot(): ToolSnapshot<TOutput, TOptions> {
    return {
      name: this.name,
      description: this.description,
      cache: this.cache,
      options: shallowCopy(this.options),
      emitter: this.emitter,
    };
  }

  loadSnapshot(snapshot: ToolSnapshot<TOutput, TOptions>): void {
    Object.assign(this, snapshot);
  }

  pipe<S extends AnyTool, T extends AnyTool>(
    this: S,
    tool: T,
    mapper: (
      input: ToolInputRaw<S>,
      output: TOutput,
      options: Partial<TRunOptions>,
      run: RunContext<
        DynamicTool<TOutput, ZodSchema<ToolInput<S>>, TOptions, TRunOptions, ToolInput<S>>
      >,
    ) => ToolInputRaw<typeof tool>,
  ) {
    return new DynamicTool<TOutput, ZodSchema<ToolInput<S>>, TOptions, TRunOptions, ToolInput<S>>({
      name: this.name,
      description: this.description,
      options: this.options,
      inputSchema: this.inputSchema() as ZodSchema<ToolInput<S>>,
      handler: async (input: ToolInputRaw<S>, options, run): Promise<TOutput> => {
        const selfOutput = await this.run(input, options);
        const wrappedInput = mapper(input, selfOutput, options, run);
        return await tool.run(wrappedInput);
      },
    } as const);
  }

  extend<S extends AnyTool, TS extends ZodSchema>(
    this: S,
    schema: TS,
    mapper: (
      input: z.output<TS>,
      options: Partial<TRunOptions>,
      run: RunContext<DynamicTool<TOutput, TS, TOptions, TRunOptions, z.output<TS>>>,
    ) => ToolInputRaw<S>,
    overrides: {
      name?: string;
      description?: string;
    } = {},
  ) {
    return new DynamicTool<TOutput, TS, TOptions, TRunOptions, z.output<TS>>({
      name: overrides?.name || this.name,
      description: overrides?.name || this.description,
      options: shallowCopy(this.options),
      inputSchema: schema,
      handler: async (input: ToolInputRaw<S>, options, run): Promise<TOutput> => {
        const wrappedInput = mapper(input, options, run);
        return await this.run(wrappedInput, options);
      },
    } as const);
  }
}

export type AnyTool = Tool<any, any, any>;

export class DynamicTool<
  TOutput extends ToolOutput,
  TInputSchema extends AnyToolSchemaLike,
  TOptions extends BaseToolOptions = BaseToolOptions,
  TRunOptions extends BaseToolRunOptions = BaseToolRunOptions,
  TInput = FromSchemaLike<TInputSchema>,
> extends Tool<TOutput, TOptions, TRunOptions> {
  static {
    this.register();
  }

  declare name: string;
  declare description: string;
  private readonly _inputSchema: TInputSchema;
  declare readonly emitter: Emitter<ToolEvents<FromSchemaLike<TInputSchema>, TOutput>>;
  private readonly handler;

  inputSchema(): TInputSchema {
    return this._inputSchema;
  }

  constructor(fields: {
    name: string;
    description: string;
    inputSchema: TInputSchema;
    handler: (
      input: TInput,
      options: Partial<TRunOptions>,
      run: GetRunContext<DynamicTool<TOutput, TInputSchema, TOptions, TRunOptions, TInput>>,
    ) => Promise<TOutput>;
    options?: TOptions;
  }) {
    validate(
      fields,
      z.object({
        name: z
          .string({ message: "Tool must have a name" })
          .refine((val) => /^[a-zA-Z0-9\-_]+$/.test(val), {
            message: "Tool name must only have -, _, letters or numbers",
          }),
        description: z
          .string({ message: "Tool must have a description" })
          .refine((val) => val && val !== "", { message: "Tool must have a description" }),
        inputSchema: z.union([z.instanceof(ZodSchema), z.object({}).passthrough()], {
          message: "Tool must have a schema",
        }),
        handler: z.function(),
        options: z.object({}).passthrough().optional(),
      }),
    );
    super(...([fields.options] as ToolConstructorParameters<TOptions>));
    this.name = fields.name;
    this.description = fields.description;
    this._inputSchema = fields.inputSchema;
    this.handler = fields.handler;
    this.emitter = Emitter.root.child({
      namespace: ["tool", "dynamic", toCamelCase(this.name)],
      creator: this,
    });
  }

  protected _run(
    arg: TInput,
    options: Partial<TRunOptions>,
    run: GetRunContext<DynamicTool<TOutput, TInputSchema, TOptions, TRunOptions, TInput>>,
  ): Promise<TOutput> {
    return this.handler(arg, options, run);
  }

  createSnapshot() {
    return { ...super.createSnapshot(), handler: this.handler, _inputSchema: this._inputSchema };
  }

  loadSnapshot({ handler, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    super.loadSnapshot(snapshot);
    Object.assign(this, { handler });
  }
}
