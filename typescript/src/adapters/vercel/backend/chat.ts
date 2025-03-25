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
  CoreMessage,
  CoreToolMessage,
  generateObject,
  generateText,
  jsonSchema,
  LanguageModelV1,
  streamText,
  TextPart,
  ToolCallPart,
} from "ai";
import { Emitter } from "@/emitter/emitter.js";
import {
  AssistantMessage,
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
import { ZodArray, ZodEnum } from "zod";

export abstract class VercelChatModel<
  M extends LanguageModelV1 = LanguageModelV1,
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

  protected async _create(input: ChatModelInput, _run: GetRunContext<this>) {
    const {
      finishReason,
      usage,
      response: { messages },
    } = await generateText(await this.transformInput(input));

    return new ChatModelOutput(this.transformMessages(messages), usage, finishReason);
  }

  protected async _createStructure<T>(
    { schema, ...input }: ChatModelObjectInput<T>,
    run: GetRunContext<this>,
  ): Promise<ChatModelObjectOutput<T>> {
    const inputSchema = schema._input || schema;

    const response = await generateObject({
      temperature: 0,
      ...(await this.transformInput(input)),
      schema,
      abortSignal: run.signal,
      model: this.model,
      // @ts-expect-error
      output:
        inputSchema instanceof ZodArray
          ? "array"
          : inputSchema instanceof ZodEnum
            ? "enum"
            : "object",
    });

    return { object: response.object };
  }

  async *_createStream(input: ChatModelInput, run: GetRunContext<this>) {
    if (!this.supportsToolStreaming && !isEmpty(input.tools ?? [])) {
      const response = await this._create(input, run);
      yield response;
      return;
    }

    const { fullStream, usage, finishReason, response } = streamText({
      ...(await this.transformInput(input)),
      abortSignal: run.signal,
    });

    let lastChunk: ChatModelOutput | null = null;
    for await (const event of fullStream) {
      let message: Message;
      switch (event.type) {
        case "text-delta":
          message = new AssistantMessage(event.textDelta);
          break;
        case "tool-call":
          message = new AssistantMessage({
            type: event.type,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            args: event.args,
          });
          break;
        case "error":
          throw new ChatModelError("Unhandled error", [event.error as Error]);
        case "step-finish":
        case "step-start":
          continue;
        case "tool-result":
          message = new ToolMessage({
            type: event.type,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            result: event.result,
          });
          break;
        case "tool-call-streaming-start":
        case "tool-call-delta":
          continue;
        case "finish":
          continue;
        default:
          throw new Error(`Unhandled event "${event.type}"`);
      }
      lastChunk = new ChatModelOutput([message]);
      yield lastChunk;
    }

    if (!lastChunk) {
      throw new ChatModelError("No chunks have been received!");
    }
    lastChunk.usage = await usage;
    lastChunk.finishReason = await finishReason;
    await response;
  }

  protected async transformInput(
    input: ChatModelInput,
  ): Promise<Parameters<typeof generateText<Record<string, any>>>[0]> {
    const tools = await Promise.all(
      (input.tools ?? []).map(async (tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: jsonSchema(await tool.getInputJsonSchema()),
      })),
    );

    const messages = input.messages.map((msg): CoreMessage => {
      if (msg instanceof AssistantMessage) {
        return { role: "assistant", content: msg.content };
      } else if (msg instanceof ToolMessage) {
        return { role: "tool", content: msg.content };
      } else if (msg instanceof UserMessage) {
        return { role: msg.role, content: msg.content };
      } else if (msg instanceof SystemMessage) {
        return { role: "user", content: msg.content };
      }
      return { role: msg.role, content: msg.content } as CoreMessage;
    });

    return {
      ...this.parameters,
      ...input,
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
