/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
} from "@/backend/chat.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import {
  BaseChatModel,
  BaseChatModelCallOptions,
} from "@langchain/core/language_models/chat_models";
import { AIMessageChunk } from "@langchain/core/messages";
import { toBeeAIMessages, toLCMessages } from "./utils.js";

const FINISH_REASONS: Record<string, ChatModelFinishReason> = {
  stop: "stop",
  end_turn: "stop",
  stop_sequence: "stop",
  length: "length",
  max_tokens: "length",
  model_context_window_exceeded: "length",
  tool_calls: "tool-calls",
  function_call: "tool-calls",
  tool_use: "tool-calls",
  content_filter: "content-filter",
  content_filtered: "content-filter",
  guardrail_intervened: "content-filter",
  refusal: "content-filter",
  safety: "content-filter",
  error: "error",
};

function toFinishReason(metadata?: Record<string, unknown>): ChatModelFinishReason | undefined {
  // Ollama reports "done_reason", OpenAI "finish_reason", Anthropic "stop_reason",
  // Bedrock "stopReason" and Google "finishReason".
  const raw =
    metadata?.done_reason ??
    metadata?.finish_reason ??
    metadata?.stop_reason ??
    metadata?.stopReason ??
    metadata?.finishReason;
  if (typeof raw !== "string" || !raw) {
    return undefined;
  }
  return FINISH_REASONS[raw.toLowerCase()] ?? "other";
}

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
    const messages = toLCMessages(input.messages);

    const options: BaseChatModelCallOptions = {
      runId: run.runId,
      stop: input.stopSequences,
      signal: run.signal,
      tool_choice: input.toolChoice,
    };

    return { messages, options };
  }

  protected prepareOutput(output: AIMessageChunk) {
    const messages = toBeeAIMessages([output]);

    const stop = toFinishReason(output.response_metadata);
    return new ChatModelOutput(
      messages,
      {
        totalTokens: output.usage_metadata?.total_tokens ?? 0,
        promptTokens: output.usage_metadata?.input_tokens ?? 0,
        completionTokens: output.usage_metadata?.output_tokens ?? 0,
        reasoningTokens: output.usage_metadata?.output_token_details?.reasoning,
        cachedPromptTokens: output.usage_metadata?.input_token_details?.cache_read,
      },
      stop,
    );
  }

  protected async _createStructure<T>(
    input: ChatModelObjectInput<T>,
    run: RunContext<this>,
  ): Promise<ChatModelObjectOutput<T>> {
    const { messages, options } = this.prepareInput(input, run);
    const { raw, parsed } = await this.lcLLM
      .withStructuredOutput<any>(input.schema, {
        method: "jsonSchema",
        strict: false,
        includeRaw: true,
      })
      .invoke(messages, options);

    return { object: parsed as T, output: this.prepareOutput(raw as AIMessageChunk) };
  }

  createSnapshot() {
    return { ...super.createSnapshot(), emitter: this.emitter, lcLLM: this.lcLLM };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }

  static {
    this.register();
  }
}
