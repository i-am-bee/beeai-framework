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
import { shallowCopy } from "@/serializer/utils.js";
import { customMerge } from "@/internals/helpers/object.js";
import { takeBigger } from "@/internals/helpers/number.js";
import { Callback } from "@/emitter/types.js";
import { FrameworkError } from "@/errors.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext, RunContext } from "@/context.js";
import { isFunction } from "remeda";
import { ObjectHashKeyFn } from "@/cache/decoratorCache.js";
import { Task } from "promise-based-task";
import { NullCache } from "@/cache/nullCache.js";
import { BaseCache } from "@/cache/base.js";
import { FullModelName, loadModel, parseModel } from "@/backend/utils.js";
import { ProviderName } from "@/backend/constants.js";
import { AnyTool } from "@/tools/base.js";
import { AssistantMessage, Message, SystemMessage, UserMessage } from "@/backend/message.js";
import {
  JSONSchema7,
  LanguageModelV1FunctionTool,
  LanguageModelV1ProviderDefinedTool,
  LanguageModelV1ToolChoice,
} from "@ai-sdk/provider";
import { ChatModelError } from "@/backend/errors.js";
import { z } from "zod";
import {
  createSchemaValidator,
  parseBrokenJson,
  toJsonSchema,
} from "@/internals/helpers/schema.js";
import { Retryable } from "@/internals/helpers/retryable.js";
import type { ValidateFunction } from "ajv";
import { PromptTemplate } from "@/template.js";
import { toAsyncGenerator } from "@/internals/helpers/promise.js";
import { Serializer } from "@/serializer/serializer.js";

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

export interface ChatModelObjectInput<T> extends ChatModelParameters {
  schema: z.ZodSchema<T>;
  messages: Message[];
  abortSignal?: AbortSignal;
  maxRetries?: number;
}

export interface ChatModelObjectOutput<T> {
  object: T;
}

export interface ChatModelInput extends ChatModelParameters {
  tools?: AnyTool[];
  abortSignal?: AbortSignal;
  stopSequences?: string[];
  responseFormat?:
    | {
        type: "regular";
        tools?: (LanguageModelV1FunctionTool | LanguageModelV1ProviderDefinedTool)[];
        toolChoice?: LanguageModelV1ToolChoice;
      }
    | {
        type: "object-json";
        schema?: JSONSchema7;
        name?: string;
        description?: string;
      }
    | {
        type: "object-tool";
        tool: LanguageModelV1FunctionTool;
      };
  toolChoice?: never; // TODO
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

export abstract class ChatModel extends Serializable {
  public abstract readonly emitter: Emitter<ChatModelEvents>;
  public cache: ChatModelCache = new NullCache();
  public parameters: ChatModelParameters = {};

  abstract get modelId(): string;
  abstract get providerId(): string;

  create(input: ChatModelInput & { stream?: boolean }) {
    input = shallowCopy(input);

    return RunContext.enter(
      this,
      { params: [input] as const, signal: input?.abortSignal },
      async (run) => {
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

    const systemTemplate = new PromptTemplate({
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

        return { object };
      },
      config: {
        signal: run.signal,
        maxRetries: input?.maxRetries || 1,
      },
    }).get();
  }

  createSnapshot() {
    return { cache: this.cache, emitter: this.emitter, parameters: shallowCopy(this.parameters) };
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
