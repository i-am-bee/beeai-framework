/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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
import { shallowCopy } from "@/serializer/utils.js";
import { customMerge } from "@/internals/helpers/object.js";
import { takeBigger } from "@/internals/helpers/number.js";
import { Callback } from "@/emitter/types.js";
import { FrameworkError } from "@/errors.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext, RunContext } from "@/context.js";
import { isEmpty, isFunction, randomString } from "remeda";
import { ObjectHashKeyFn } from "@/cache/decoratorCache.js";
import { Task } from "promise-based-task";
import { NullCache } from "@/cache/nullCache.js";
import { BaseCache } from "@/cache/base.js";
import {
  filterToolsByToolChoice,
  FullModelName,
  generateToolUnionSchema,
  loadModel,
  parseModel,
} from "@/backend/utils.js";
import { ProviderName } from "@/backend/constants.js";
import { AnyTool, Tool } from "@/tools/base.js";
import { AssistantMessage, Message, SystemMessage, UserMessage } from "@/backend/message.js";

import { ChatModelError } from "@/backend/errors.js";
import { z, ZodSchema, ZodType } from "zod";
import {
  createSchemaValidator,
  parseBrokenJson,
  toJsonSchema,
} from "@/internals/helpers/schema.js";
import { Retryable } from "@/internals/helpers/retryable.js";
import { SchemaObject, ValidateFunction } from "ajv";
import { PromptTemplate } from "@/template.js";
import { toAsyncGenerator } from "@/internals/helpers/promise.js";
import { Serializer } from "@/serializer/serializer.js";
import { Logger } from "@/logger/logger.js";

export interface ChatModelParameters {
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  temperature?: number;
  topK?: number;
  n?: number;
  presencePenalty?: number;
  seed?: number;
  stopSequences?: string[];
}

interface ResponseObjectJson {
  type: "object-json";
  schema?: SchemaObject;
  name?: string;
  description?: string;
}

export interface ChatModelObjectInput<T> extends ChatModelParameters {
  schema: z.ZodSchema<T> | ResponseObjectJson;
  systemPromptTemplate?: PromptTemplate<ZodType<{ schema: string }>>;
  messages: Message[];
  abortSignal?: AbortSignal;
  maxRetries?: number;
}

export interface ChatModelObjectOutput<T> {
  object: T;
  output: ChatModelOutput;
}

export type ChatModelToolChoice = "auto" | "none" | "required" | AnyTool;

export interface ChatModelInput extends ChatModelParameters {
  tools?: AnyTool[];
  abortSignal?: AbortSignal;
  stopSequences?: string[];
  responseFormat?: ZodSchema | ResponseObjectJson;
  toolChoice?: ChatModelToolChoice;
  messages: Message[];
}

export type ChatModelFinishReason =
  | "stop"
  | "length"
  | "content-filter"
  | "tool-calls"
  | "error"
  | "other"
  | "unknown";

export interface ChatModelUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface ChatModelEvents {
  newToken?: Callback<{ value: ChatModelOutput; callbacks: { abort: () => void } }>;
  success?: Callback<{ value: ChatModelOutput }>;
  start?: Callback<{ input: ChatModelInput }>;
  error?: Callback<{ input: ChatModelInput; error: FrameworkError }>;
  finish?: Callback<null>;
}

export type ChatModelEmitter<A = Record<never, never>> = Emitter<
  ChatModelEvents & Omit<A, keyof ChatModelEvents>
>;

export type ChatModelCache = BaseCache<Task<ChatModelOutput[]>>;
export type ConfigFn<T> = (value: T) => T;
export interface ChatConfig {
  cache?: ChatModelCache | ConfigFn<ChatModelCache>;
  parameters?: ChatModelParameters | ConfigFn<ChatModelParameters>;
}

export type ChatModelToolChoiceSupport = "required" | "none" | "single" | "auto";

export abstract class ChatModel extends Serializable {
  public abstract readonly emitter: Emitter<ChatModelEvents>;
  public cache: ChatModelCache = new NullCache();
  public parameters: ChatModelParameters = {};
  protected readonly logger = Logger.root.child({
    name: this.constructor.name,
  });

  public readonly toolChoiceSupport: ChatModelToolChoiceSupport[] = [
    "required",
    "none",
    "single",
    "auto",
  ];
  public toolCallFallbackViaResponseFormat = true;
  public readonly modelSupportsToolCalling: boolean = true;

  abstract get modelId(): string;
  abstract get providerId(): string;

  create(input: ChatModelInput & { stream?: boolean }) {
    input = shallowCopy(input);

    return RunContext.enter(
      this,
      { params: [input] as const, signal: input?.abortSignal },
      async (run) => {
        if (!this.modelSupportsToolCalling) {
          input.tools = [];
        }

        const forceToolCallViaResponseFormat = this.shouldForceToolCallViaResponseFormat(input);
        if (forceToolCallViaResponseFormat && input.tools && !isEmpty(input.tools)) {
          input.responseFormat = await generateToolUnionSchema(
            filterToolsByToolChoice(input.tools, input.toolChoice),
          );
          input.toolChoice = undefined;
        }

        if (!this.isToolChoiceSupported(input.toolChoice)) {
          this.logger.warn(
            `The following tool choice value '${input.toolChoice}' is not supported. Ignoring.`,
          );
          input.toolChoice = undefined;
        }

        const cacheEntry = await this.createCacheAccessor(input);

        try {
          await run.emitter.emit("start", { input });
          const chunks: ChatModelOutput[] = [];

          const generator =
            cacheEntry.value ??
            (input.stream
              ? this._createStream(input, run)
              : toAsyncGenerator(this._create(input, run)));

          const controller = new AbortController();
          for await (const value of generator) {
            chunks.push(value);
            await run.emitter.emit("newToken", {
              value,
              callbacks: { abort: () => controller.abort() },
            });
            if (controller.signal.aborted) {
              break;
            }
          }

          cacheEntry.resolve(chunks);
          const result = ChatModelOutput.fromChunks(chunks);

          if (forceToolCallViaResponseFormat && isEmpty(result.getToolCalls())) {
            const lastMsg = result.messages.at(-1)!;
            const toolCall = parseBrokenJson(lastMsg.text);
            if (!toolCall) {
              throw new ChatModelError(
                `Failed to produce a valid tool call. Generate output: ${lastMsg.text}`,
                [],
                {
                  isFatal: true,
                  isRetryable: false,
                },
              );
            }
            lastMsg.content.length = 0;
            lastMsg.content.push({
              type: "tool-call",
              toolCallId: `call_${randomString(8).toLowerCase()}`,
              toolName: toolCall.name, // todo: add types
              args: toolCall.parameters,
            });
          }

          await run.emitter.emit("success", { value: result });
          return result;
        } catch (error) {
          await run.emitter.emit("error", { input, error });
          await cacheEntry.reject(error);
          if (error instanceof ChatModelError) {
            throw error;
          } else {
            throw new ChatModelError(`The Chat Model has encountered an error.`, [error]);
          }
        } finally {
          await run.emitter.emit("finish", null);
        }
      },
    );
  }

  createStructure<T>(input: ChatModelObjectInput<T>) {
    return RunContext.enter(
      this,
      { params: [input] as const, signal: input?.abortSignal },
      async (run) => {
        return await this._createStructure<T>(input, run);
      },
    );
  }

  config({ cache, parameters }: ChatConfig): void {
    if (cache) {
      this.cache = isFunction(cache) ? cache(this.cache) : cache;
    }
    if (parameters) {
      this.parameters = isFunction(parameters) ? parameters(this.parameters) : parameters;
    }
  }

  static async fromName(name: FullModelName | ProviderName, options?: ChatModelParameters) {
    const { providerId, modelId } = parseModel(name);
    const Target = await loadModel<ChatModel>(providerId, "chat");
    return new Target(modelId || undefined, options);
  }

  protected abstract _create(
    input: ChatModelInput,
    run: GetRunContext<typeof this>,
  ): Promise<ChatModelOutput>;
  protected abstract _createStream(
    input: ChatModelInput,
    run: GetRunContext<typeof this>,
  ): AsyncGenerator<ChatModelOutput, void>;

  protected async _createStructure<T>(
    input: ChatModelObjectInput<T>,
    run: GetRunContext<typeof this>,
  ): Promise<ChatModelObjectOutput<T>> {
    const { schema, ...options } = input;
    const jsonSchema = toJsonSchema(schema);

    const systemTemplate =
      input.systemPromptTemplate ??
      new PromptTemplate({
        schema: z.object({
          schema: z.string().min(1),
        }),
        template: `You are a helpful assistant that generates only valid JSON adhering to the following JSON Schema.

\`\`\`
{{schema}}
\`\`\`

IMPORTANT: You MUST answer with a JSON object that matches the JSON schema above.`,
      });

    const messages: Message[] = [
      new SystemMessage(systemTemplate.render({ schema: JSON.stringify(jsonSchema, null, 2) })),
      ...input.messages,
    ];

    const errorTemplate = new PromptTemplate({
      schema: z.object({
        errors: z.string(),
        expected: z.string(),
        received: z.string(),
      }),
      template: `Generated object does not match the expected JSON schema!

Validation Errors: {{errors}}`,
    });

    return new Retryable<ChatModelObjectOutput<T>>({
      executor: async () => {
        const response = await this._create(
          {
            ...options,
            messages,
            responseFormat: { type: "object-json" },
          },
          run,
        );

        const textResponse = response.getTextContent();
        const object: T = parseBrokenJson(textResponse, { pair: ["{", "}"] });
        const validator = createSchemaValidator(schema) as ValidateFunction<T>;

        const success = validator(object);
        if (!success) {
          const context = {
            expected: JSON.stringify(jsonSchema),
            received: textResponse,
            errors: JSON.stringify(validator.errors ?? []),
          };

          messages.push(new UserMessage(errorTemplate.render(context)));
          throw new ChatModelError(`LLM did not produce a valid output.`, [], {
            context,
          });
        }

        return {
          object,
          output: response,
        };
      },
      config: {
        signal: run.signal,
        maxRetries: input?.maxRetries || 1,
      },
    }).get();
  }

  createSnapshot() {
    return {
      cache: this.cache,
      emitter: this.emitter,
      parameters: shallowCopy(this.parameters),
      logger: this.logger,
      toolChoiceSupport: this.toolChoiceSupport.slice(),
      toolCallFallbackViaResponseFormat: this.toolCallFallbackViaResponseFormat,
      modelSupportsToolCalling: this.modelSupportsToolCalling,
    };
  }

  destroy() {
    this.emitter.destroy();
  }

  protected async createCacheAccessor({
    abortSignal: _,
    messages,
    tools = [],
    ...input
  }: ChatModelInput) {
    const key = ObjectHashKeyFn({
      ...input,
      messages: await Serializer.serialize(messages.map((msg) => msg.toPlain())),
      tools: await Serializer.serialize(tools),
    });
    const value = await this.cache.get(key);
    const isNew = value === undefined;

    let task: Task<ChatModelOutput[]> | null = null;
    if (isNew) {
      task = new Task();
      await this.cache.set(key, task);
    }

    return {
      key,
      value,
      resolve: <T2 extends ChatModelOutput>(value: T2[]) => {
        task?.resolve?.(value);
      },
      reject: async (error: Error) => {
        task?.reject?.(error);
        if (isNew) {
          await this.cache.delete(key);
        }
      },
    };
  }

  protected shouldForceToolCallViaResponseFormat({
    tools = [],
    toolChoice,
    responseFormat,
  }: ChatModelInput) {
    if (
      isEmpty(tools) ||
      !toolChoice ||
      toolChoice === "none" ||
      toolChoice === "auto" ||
      !this.toolCallFallbackViaResponseFormat ||
      Boolean(responseFormat)
    ) {
      return false;
    }

    const toolChoiceSupported = this.isToolChoiceSupported(toolChoice);
    return !this.modelSupportsToolCalling || !toolChoiceSupported;
  }

  protected isToolChoiceSupported(choice?: ChatModelToolChoice): boolean {
    return (
      !choice ||
      (choice instanceof Tool
        ? this.toolChoiceSupport.includes("single")
        : this.toolChoiceSupport.includes(choice))
    );
  }
}

export class ChatModelOutput extends Serializable {
  constructor(
    public readonly messages: Message[],
    public usage?: ChatModelUsage,
    public finishReason?: ChatModelFinishReason,
  ) {
    super();
  }

  static fromChunks(chunks: ChatModelOutput[]) {
    const final = new ChatModelOutput([]);
    chunks.forEach((cur) => final.merge(cur));
    return final;
  }

  merge(other: ChatModelOutput) {
    this.messages.push(...other.messages);
    this.finishReason = other.finishReason;
    if (this.usage && other.usage) {
      this.usage = customMerge([this.usage, other.usage], {
        totalTokens: takeBigger,
        promptTokens: takeBigger,
        completionTokens: takeBigger,
      });
    } else if (other.usage) {
      this.usage = shallowCopy(other.usage);
    }
  }

  getToolCalls() {
    return this.messages
      .filter((r) => r instanceof AssistantMessage)
      .flatMap((r) => r.getToolCalls())
      .filter(Boolean);
  }

  getTextMessages(): AssistantMessage[] {
    return this.messages.filter((r) => r instanceof AssistantMessage).filter((r) => r.text);
  }

  getTextContent(): string {
    return this.messages
      .filter((r) => r instanceof AssistantMessage)
      .flatMap((r) => r.text)
      .filter(Boolean)
      .join("");
  }

  toString() {
    return this.getTextContent();
  }

  createSnapshot() {
    return {
      messages: shallowCopy(this.messages),
      usage: shallowCopy(this.usage),
      finishReason: this.finishReason,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
