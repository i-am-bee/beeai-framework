/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "@/internals/serializable.js";
import { shallowCopy } from "@/serializer/utils.js";
import { customMerge, getLast } from "@/internals/helpers/object.js";
import { takeBigger } from "@/internals/helpers/number.js";
import { Callback } from "@/emitter/types.js";
import { FrameworkError } from "@/errors.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext, MiddlewareType, RunContext } from "@/context.js";
import { isEmpty, isFunction, isPromise, isString, randomString } from "remeda";
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

import {
  ChatModelError,
  ChatModelToolCallError,
  EmptyChatModelResponseError,
} from "@/backend/errors.js";
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
import { ToolCallPart } from "ai";
import { isToolCallValid } from "@/adapters/vercel/backend/utils.js";

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
  stream?: boolean;
}

interface ResponseObjectJson {
  type: "object-json";
  schema?: SchemaObject;
  name?: string;
  description?: string;
}

export interface ChatModelObjectInput<T> extends ChatModelParameters {
  schema: z.ZodSchema<T, any, any> | ResponseObjectJson;
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
  streamPartialToolCalls?: boolean;
  maxRetries?: number;
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
  reasoningTokens?: number;
  cachedPromptTokens?: number;
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
  public readonly middlewares: MiddlewareType<typeof this>[] = [];
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
  public readonly fixInvalidToolCalls: boolean = true;
  public readonly retryOnEmptyResponse: boolean = true;

  abstract get modelId(): string;
  abstract get providerId(): string;

  create(input: ChatModelInput) {
    input = shallowCopy(input);
    if (input.stream === undefined) {
      input.stream = this.parameters.stream;
    }

    return RunContext.enter(
      this,
      { params: [input] as const, signal: input?.abortSignal },
      async (run): Promise<ChatModelOutput> => {
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

        const modelInputMessagesBackup = input.messages.slice();
        const cacheEntry = await this.createCacheAccessor(input);

        try {
          await run.emitter.emit("start", { input });

          const result = await new Retryable({
            executor: async () => {
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

              const result = ChatModelOutput.fromChunks(chunks);
              if (result.isEmpty()) {
                throw new EmptyChatModelResponseError();
              }

              if (
                isEmpty(result.getToolCalls()) &&
                (forceToolCallViaResponseFormat ||
                  input.toolChoice === "required" ||
                  input.toolChoice instanceof Tool)
              ) {
                const lastMsg = result.messages.at(-1)!;
                let toolCall = parseBrokenJson(lastMsg.text, { pair: ["{", "}"] });
                if (
                  toolCall &&
                  !toolCall.name &&
                  !toolCall.parameters &&
                  input.toolChoice instanceof Tool
                ) {
                  toolCall = { name: input.toolChoice.name, parameters: toolCall };
                }
                if (!toolCall || !toolCall.name || !toolCall.parameters) {
                  throw new ChatModelToolCallError(
                    `Failed to produce a valid tool call. Generate output: ${lastMsg.text}`,
                    [],
                    {
                      generatedContent: lastMsg.text,
                      generatedError: "Tool call was not produced.",
                      response: result,
                    },
                  );
                }
                lastMsg.content.length = 0;
                lastMsg.content.push({
                  type: "tool-call",
                  toolCallId: `call_${randomString(8).toLowerCase()}`,
                  toolName: toolCall.name, // todo: add types
                  input: toolCall.parameters,
                });
              }

              for (const toolCall of result.getToolCalls()) {
                const tool = input.tools?.find((t) => t.name === toolCall.toolName);
                if (!tool) {
                  const availableTools = input.tools?.map((t) => t.name).join(",") || "None";
                  throw new ChatModelToolCallError("Non existing tool call.", [], {
                    generatedError: `Error: Unknown tool '${toolCall.toolName}'.\nUse on of the available tools: ${availableTools}`,
                    generatedContent: JSON.stringify({
                      name: toolCall.toolName,
                      input: isString(toolCall.input)
                        ? toolCall.input
                        : JSON.stringify(toolCall.input),
                    }),
                    response: result,
                  });
                }

                if (!isToolCallValid(toolCall)) {
                  throw new ChatModelToolCallError("Malformed tool call.", [], {
                    generatedContent: isString(toolCall.input)
                      ? toolCall.input
                      : JSON.stringify(toolCall.input),
                    generatedError: `The tool call for the '${toolCall.toolName}' tool has malformed parameters. It must be a valid JSON.`,
                    response: result,
                  });
                }

                if (isString(toolCall.input)) {
                  toolCall.input = JSON.parse(toolCall.input);
                }
              }

              cacheEntry.resolve(chunks);

              if (!result.finishReason) {
                if (result.getToolCalls().length > 0) {
                  result.finishReason = "tool-calls";
                }
              }

              return result;
            },
            config: { maxRetries: input.maxRetries ?? 0 },
            onRetry: async (_, lastError) => {
              if (this.fixInvalidToolCalls && lastError instanceof ChatModelToolCallError) {
                input.messages = input.messages.slice();
                if (lastError.data.generatedContent) {
                  input.messages.push(
                    new AssistantMessage(lastError.data.generatedContent, {
                      tempMessage: true,
                    }),
                  );
                }

                const toolNames = input.tools?.map((t) => t.name).join(", ") || "None";
                input.messages.push(
                  new UserMessage(
                    `${lastError.data.generatedError}\n\nAvailable Tools: ${toolNames}`,
                    {
                      tempMessage: true,
                    },
                  ),
                );
              } else if (
                this.retryOnEmptyResponse &&
                lastError instanceof EmptyChatModelResponseError
              ) {
                input.messages = input.messages.slice();
                const lastMessage = input.messages.at(-1);
                if (
                  lastMessage &&
                  lastMessage instanceof AssistantMessage &&
                  lastMessage.meta["tempMessage"] &&
                  lastMessage.text === ""
                ) {
                  input.messages.push(
                    new UserMessage(
                      "No output received. Please regenerate your previous response.",
                      { tempMessage: true },
                    ),
                  );
                } else {
                  // Python compatibility
                  input.messages.push(new AssistantMessage("", { tempMessage: true }));
                }
              }
            },
          }).get();

          input.messages = modelInputMessagesBackup;
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
    ).middleware(...this.middlewares);
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
    const instance = new Target(modelId || undefined);
    if (options) {
      Object.assign(instance.parameters, options);
    }
    return instance;
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
      middlewares: shallowCopy(this.middlewares) as MiddlewareType<any>[],
      parameters: shallowCopy(this.parameters),
      logger: this.logger,
      toolChoiceSupport: this.toolChoiceSupport.slice(),
      toolCallFallbackViaResponseFormat: this.toolCallFallbackViaResponseFormat,
      modelSupportsToolCalling: this.modelSupportsToolCalling,
      retryOnEmptyResponse: this.retryOnEmptyResponse,
      fixInvalidToolCalls: this.fixInvalidToolCalls,
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
    this.dedupe();
  }

  static fromChunks(chunks: ChatModelOutput[]) {
    const final = new ChatModelOutput([]);
    chunks.forEach((cur) => final.merge(cur));
    return final;
  }

  isEmpty() {
    if (this.messages.length === 0) {
      return true;
    }
    return this.getTextContent() === "" && this.getToolCalls().length === 0;
  }

  merge(other: ChatModelOutput) {
    if (other.messages.length > 0) {
      const clones = other.messages.map(cloneSync);
      this.messages.push(...clones);
      this.dedupe();
    }

    this.finishReason = other.finishReason;
    if (this.usage && other.usage) {
      this.usage = customMerge([this.usage, other.usage], {
        totalTokens: takeBigger,
        promptTokens: takeBigger,
        completionTokens: takeBigger,
        cachedPromptTokens: takeBigger,
        reasoningTokens: takeBigger,
      });
    } else if (other.usage) {
      this.usage = shallowCopy(other.usage);
    }
  }

  dedupe(): void {
    // Dedupe messages
    if (this.messages.length > 1) {
      const messagesById = new Map<string, Message[]>();
      const messagesByToolCallId = new Map<string, AssistantMessage>();

      for (const msg of this.messages) {
        const msgId = msg.id || "";

        if (msg instanceof AssistantMessage && msg.getToolCalls().length > 0) {
          const filteredChunks: AssistantMessage["content"] = [];
          for (const chunk of msg.content) {
            if (chunk.type !== "tool-call") {
              filteredChunks.push(chunk);
              continue;
            }

            // Assume tool calls with no id refer to the most recent tool call
            if (!chunk.toolCallId && messagesByToolCallId.size > 0) {
              const lastToolCallId = getLast(messagesByToolCallId.keys(), "");
              if (lastToolCallId) {
                chunk.toolCallId = lastToolCallId;
              }
            }

            if (chunk.toolCallId && messagesByToolCallId.has(chunk.toolCallId)) {
              messagesByToolCallId.get(chunk.toolCallId)!.content.push(chunk);
            } else if (chunk.toolCallId) {
              messagesByToolCallId.set(chunk.toolCallId, msg);
              filteredChunks.push(chunk);
            }
          }

          msg.content.length = 0;
          msg.content.push(...filteredChunks);

          if (filteredChunks.length === 0) {
            continue; // nothing to process
          }
        }

        if (!messagesById.has(msgId)) {
          messagesById.set(msgId, [msg]);
        } else {
          messagesById.get(msgId)!.push(msg);
        }
      }

      this.messages.length = 0;

      for (const messages of messagesById.values()) {
        const main = messages.shift()!;
        for (const other of messages) {
          main.merge(other);
        }
        this.messages.push(main);
      }
    }

    // Dedupe tool calls
    for (const msg of this.messages) {
      if (!(msg instanceof AssistantMessage)) {
        continue;
      }

      const finalToolCalls: Record<string, ToolCallPart> = {};
      let lastId = "";

      const excludedIndexes: number[] = [];
      msg.content.forEach((chunk, index) => {
        if (chunk.type !== "tool-call") {
          return;
        }
        const id = chunk.toolCallId || lastId;
        if (!(id in finalToolCalls)) {
          finalToolCalls[id] = shallowCopy(chunk);
          msg.content[index] = finalToolCalls[id];
        } else {
          excludedIndexes.push(index);
          const lastToolCall = finalToolCalls[id];
          if (isString(lastToolCall.input) && isString(chunk.input)) {
            lastToolCall.input += chunk.input;
          } else {
            throw new Error("Chunks cannot be merged.");
          }
          if (!lastToolCall.toolName) {
            lastToolCall.toolName = chunk.toolName;
          }
        }
        lastId = id;
      });
      excludedIndexes.reverse().forEach((index) => {
        msg.content.splice(index, 1);
      });
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

function cloneSync<T extends Serializable>(serializable: T): T {
  const snapshot = serializable.createSnapshot();
  if (isPromise(snapshot)) {
    throw new Error(`createSnapshot cannot be async`);
  }

  const target = Object.create(serializable.constructor.prototype) as T;
  const load = target.loadSnapshot(snapshot);
  if (isPromise(load)) {
    throw new Error(`loadSnapshot cannot be async`);
  }

  return target;
}
