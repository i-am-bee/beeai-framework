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
  ChatModel,
  ChatModelEmitter,
  ChatModelFinishReason,
  ChatModelInput,
  ChatModelObjectInput,
  ChatModelObjectOutput,
  ChatModelOutput,
  ChatModelParameters,
  ChatModelUsage,
} from "@/backend/chat.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import {
  BaseChatModel,
  BaseChatModelCallOptions,
} from "@langchain/core/language_models/chat_models";
import { AIMessageChunk, BaseMessageLike } from "@langchain/core/messages";
import { AssistantMessage, Message } from "@/backend/message.js";
import { ValueError } from "@/errors.js";

export class LangChainChatModel extends ChatModel {
  public readonly emitter: ChatModelEmitter;

  constructor(
    protected readonly lcLLM: BaseChatModel,
    public readonly parameters: ChatModelParameters = {},
  ) {
    super();
    this.emitter = Emitter.root.child({
      namespace: ["backend", "langchain", "chat"],
      creator: this,
    });
  }

  get modelId(): string {
    return this.lcLLM._modelType();
  }

  get providerId() {
    return "langchain";
  }

  protected async _create(input: ChatModelInput, run: RunContext<this>): Promise<ChatModelOutput> {
    const preparedInput = this.prepareInput(input, run);
    const response = this.lcLLM.bindTools
      ? await this.lcLLM
          .bindTools(input.tools ?? [])
          .invoke(preparedInput.messages, preparedInput.options)
      : await this.lcLLM.invoke(preparedInput.messages, preparedInput.options);

    return this.prepareOutput(response);
  }

  protected async *_createStream(
    input: ChatModelInput,
    run: RunContext<this>,
  ): AsyncGenerator<ChatModelOutput> {
    const preparedInput = this.prepareInput(input, run);

    const stream = this.lcLLM.bindTools
      ? await this.lcLLM
          .bindTools(input.tools ?? [])
          .stream(preparedInput.messages, preparedInput.options)
      : await this.lcLLM.stream(preparedInput.messages, preparedInput.options);

    for await (const response of stream) {
      const chunk = this.prepareOutput(response);
      yield chunk;
    }
  }

  protected prepareInput(input: ChatModelInput, run: RunContext<this>) {
    const messages: BaseMessageLike[] = input.messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
      type: msg.role,
      // TODO
    }));

    const options: BaseChatModelCallOptions = {
      runId: run.runId,
      stop: input.stopSequences,
      signal: run.signal,
      tool_choice: input.toolChoice,
    };

    return { messages, options };
  }

  protected prepareOutput(output: AIMessageChunk) {
    const messages: Message[] = [];
    if (typeof output.content === "string") {
      messages.push(new AssistantMessage(output.content));
    } else {
      messages.push(
        new AssistantMessage(
          output.content.map((message) => {
            if (message.type === "text") {
              return { type: "text", text: message.text };
            } else if (message.type === "image_url") {
              return { type: "text", text: message.image_url.toString() };
            } else {
              throw new ValueError(`Unknown message type "${message.type}"`);
            }
          }),
        ),
      );
    }

    const usage: ChatModelUsage = {
      totalTokens: output.usage_metadata?.total_tokens ?? 0,
      promptTokens: output.usage_metadata?.input_tokens ?? 0,
      completionTokens: output.usage_metadata?.output_tokens ?? 0,
    };

    const stop: ChatModelFinishReason = output.response_metadata.stop_sequence || "stop";

    return new ChatModelOutput(messages, usage, stop);
  }

  protected async _createStructure<T>(
    input: ChatModelObjectInput<T>,
    run: RunContext<this>,
  ): Promise<ChatModelObjectOutput<T>> {
    const { messages, options } = this.prepareInput(input, run);
    const object = this.lcLLM
      .withStructuredOutput<any>(input.schema, {
        method: "jsonSchema",
        strict: false,
      })
      .invoke(messages, options);

    return { object: object as T };
  }

  createSnapshot() {
    return { ...super.createSnapshot(), emitter: this.emitter, lcLLM: this.lcLLM };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }
}
