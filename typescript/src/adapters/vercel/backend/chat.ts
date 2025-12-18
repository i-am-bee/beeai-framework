/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ChatModelInput,
  ChatModel,
  ChatModelOutput,
  ChatModelEvents,
  ChatModelObjectInput,
  ChatModelObjectOutput,
} from "@/backend/chat.js";
import {
  CoreAssistantMessage,
  ModelMessage,
  CoreToolMessage,
  generateObject,
  generateText,
  jsonSchema,
  LanguageModel as _LanguageModel,
  streamText,
  TextPart,
  ToolCallPart,
  ToolChoice,
} from "ai";
type LanguageModelV2 = Exclude<_LanguageModel, string>;
import { Emitter } from "@/emitter/emitter.js";
import {
  AssistantMessage,
  CustomMessage,
  Message,
  SystemMessage,
  ToolMessage,
  UserMessage,
} from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { ValueError } from "@/errors.js";
import { isEmpty, mapToObj, toCamelCase } from "remeda";
import { FullModelName } from "@/backend/utils.js";
import { ChatModelError } from "@/backend/errors.js";
import { z, ZodArray, ZodEnum, ZodSchema } from "zod";
import { Tool } from "@/tools/base.js";
import { encodeCustomMessage, extractTokenUsage } from "@/adapters/vercel/backend/utils.js";

export abstract class VercelChatModel<
  M extends LanguageModelV2 = LanguageModelV2,
> extends ChatModel {
  public readonly emitter: Emitter<ChatModelEvents>;
  public readonly supportsToolStreaming: boolean = true;

  constructor(private readonly model: M) {
    super();
    if (!this.modelId) {
      throw new ValueError("No modelId has been provided!");
    }
    this.emitter = Emitter.root.child({
      namespace: ["backend", this.providerId, "chat"],
      creator: this,
    });
  }

  get modelId(): string {
    return this.model.modelId;
  }

  get providerId(): string {
    const provider = this.model.provider.split(".")[0].split("-")[0];
    return toCamelCase(provider);
  }

  protected async _create(input: ChatModelInput, run: GetRunContext<this>) {
    const responseFormat = input.responseFormat;
    if (responseFormat && (responseFormat instanceof ZodSchema || responseFormat.schema)) {
      const { output } = await this._createStructure(
        {
          ...input,
          schema: responseFormat,
        },
        run,
      );
      return output;
    }

    const {
      finishReason,
      usage,
      response: { messages },
    } = await generateText(await this.transformInput(input));

    return new ChatModelOutput(
      this.transformMessages(messages),
      extractTokenUsage(usage),
      finishReason,
    );
  }

  protected async _createStructure<T>(
    { schema, ...input }: ChatModelObjectInput<T>,
    run: GetRunContext<this>,
  ): Promise<ChatModelObjectOutput<T>> {
    const response = await generateObject({
      temperature: 0,
      ...(await this.transformInput(input)),
      abortSignal: run.signal,
      model: this.model,
      ...(schema instanceof ZodSchema
        ? {
            schema,
            output: ((schema._input || schema) instanceof ZodArray
              ? "array"
              : (schema._input || schema) instanceof ZodEnum
                ? "enum"
                : "object") as any,
          }
        : {
            schema: schema.schema ? jsonSchema<T>(schema.schema) : z.any(),
            schemaName: schema.name,
            schemaDescription: schema.description,
          }),
    });

    return {
      object: response.object as T,
      output: new ChatModelOutput(
        [new AssistantMessage(JSON.stringify(response.object, null, 2))],
        extractTokenUsage(response.usage),
        response.finishReason,
      ),
    };
  }

  async *_createStream(input: ChatModelInput, run: GetRunContext<this>) {
    if (!this.supportsToolStreaming && !isEmpty(input.tools ?? [])) {
      const response = await this._create(input, run);
      yield response;
      return;
    }

    const {
      fullStream,
      usage: usagePromise,
      finishReason: finishReasonPromise,
      response: responsePromise,
    } = streamText({
      ...(await this.transformInput(input)),
      abortSignal: run.signal,
    });

    // TODO: handle chunk merging

    let streamEmpty = true;
    const streamedToolCalls = new Map<string, ToolCallPart>();
    for await (const event of fullStream) {
      streamEmpty = false;

      let message: Message;
      switch (event.type) {
        case "text-delta":
          message = new AssistantMessage(event.text);
          yield new ChatModelOutput([message]);
          break;
        case "tool-input-start": {
          if (!input.streamPartialToolCalls) {
            break;
          }

          const chunk: ToolCallPart = {
            type: "tool-call",
            toolName: event.toolName,
            toolCallId: event.id,
            input: "",
          };
          streamedToolCalls.set(event.id, chunk);
          const message = new AssistantMessage(chunk);
          yield new ChatModelOutput([message]);
          break;
        }
        case "tool-input-delta": {
          const chunk = streamedToolCalls.get(event.id)!;
          if (chunk && event.delta) {
            chunk.input += event.delta;
            const message = new AssistantMessage({ ...chunk, input: event.delta });
            yield new ChatModelOutput([message]);
          }
          break;
        }
        case "tool-call": {
          break;
          /*console.info("final", { chunk: event });
          const existingToolCall = streamedToolCalls.get(event.toolCallId);
          if (existingToolCall) {
            streamedToolCalls.delete(event.toolCallId);
            break;
          }
          message = new AssistantMessage({
            type: event.type,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            input: event.input,
          });
          yield new ChatModelOutput([message]);
          break;*/
        }
        case "error":
          throw new ChatModelError("Unhandled error", [event.error as Error]);
        case "tool-result":
          message = new ToolMessage({
            type: event.type,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            output: event.output as any,
          });
          yield new ChatModelOutput([message]);
          break;
        case "abort":
          break;
        default:
          break;
      }
    }

    if (streamEmpty) {
      throw new ChatModelError("No chunks have been received!");
    }

    try {
      const [usage, finishReason, _] = await Promise.all([
        usagePromise,
        finishReasonPromise,
        responsePromise,
      ]);
      const lastChunk = new ChatModelOutput([new AssistantMessage("")]);
      lastChunk.usage = extractTokenUsage(usage);
      lastChunk.finishReason = finishReason;
      yield lastChunk;
    } catch (e) {
      if (!run.signal.aborted) {
        throw e;
      }
    }
  }

  protected async transformInput(
    input: ChatModelInput,
  ): Promise<Parameters<typeof generateText<Record<string, any>>>[0]> {
    const tools = await Promise.all(
      (input.tools ?? []).map(async (tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: jsonSchema(await tool.getInputJsonSchema()),
      })),
    );

    const messages = input.messages.map((msg): ModelMessage => {
      if (msg instanceof CustomMessage) {
        msg = encodeCustomMessage(msg);
      }

      if (msg instanceof AssistantMessage) {
        return { role: "assistant", content: msg.content };
      } else if (msg instanceof ToolMessage) {
        return { role: "tool", content: msg.content };
      } else if (msg instanceof UserMessage) {
        return { role: "user", content: msg.content };
      } else if (msg instanceof SystemMessage) {
        return { role: "system", content: msg.content.map((part) => part.text).join("\n") };
      }
      return { role: msg.role, content: msg.content } as ModelMessage;
    });

    let toolChoice: ToolChoice<Record<string, any>> | undefined;
    if (input.toolChoice && input.toolChoice instanceof Tool) {
      if (this.toolChoiceSupport.includes("single")) {
        toolChoice = {
          type: "tool",
          toolName: input.toolChoice.name,
        };
      } else {
        this.logger.warn(`The single tool choice is not supported.`);
      }
    } else if (input.toolChoice) {
      if (this.toolChoiceSupport.includes(input.toolChoice)) {
        toolChoice = input.toolChoice;
      } else {
        this.logger.warn(`The following tool choice value '${input.toolChoice}' is not supported.`);
      }
    }

    return {
      ...this.parameters,
      ...input,
      toolChoice,
      model: this.model,
      tools: mapToObj(tools, ({ name, ...tool }) => [name, tool]),
      messages,
    };
  }

  protected transformMessages(messages: (CoreAssistantMessage | CoreToolMessage)[]): Message[] {
    return messages.flatMap((msg) => {
      if (msg.role === "tool") {
        return new ToolMessage(msg.content, msg.providerOptions);
      }
      return new AssistantMessage(
        msg.content as TextPart | ToolCallPart | string,
        msg.providerOptions,
      );
    });
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      providerId: this.providerId,
      modelId: this.modelId,
      supportsToolStreaming: this.supportsToolStreaming,
    };
  }

  async loadSnapshot({ providerId, modelId, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    const instance = await ChatModel.fromName(`${providerId}:${modelId}` as FullModelName);
    if (!(instance instanceof VercelChatModel)) {
      throw new Error("Incorrect deserialization!");
    }
    instance.destroy();
    Object.assign(this, {
      ...snapshot,
      model: instance.model,
    });
  }
}
