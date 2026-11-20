/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { AnthropicClient, AnthropicClientSettings } from "@/adapters/anthropic/backend/client.js";
import { getEnv } from "@/internals/env.js";
import { AnthropicProvider } from "@ai-sdk/anthropic";
import { ChatModelInput, ChatModelParameters, ChatModelToolChoiceSupport } from "@/backend/chat.js";
import type { generateText } from "ai";

type AnthropicParameters = Parameters<AnthropicProvider["languageModel"]>;
export type AnthropicChatModelId = NonNullable<AnthropicParameters[0]>;

// Claude models that answer a forced tool choice (`any` / `tool`) with a 400. The AI SDK silently
// downgrades it to `auto` for them, so ask for the tool call through structured output instead.
// Matched as a prefix so dated snapshots are covered while `claude-opus-5` / `claude-sonnet-5` are not.
const NO_FORCED_TOOL_CHOICE_PREFIXES = [
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "claude-fable-5-1",
  "claude-mythos-5-1",
];

export class AnthropicChatModel extends VercelChatModel {
  public readonly toolChoiceSupport: ChatModelToolChoiceSupport[] = this.rejectsForcedToolChoice
    ? ["none", "auto"]
    : ["required", "none", "single", "auto"];

  constructor(
    modelId: AnthropicChatModelId = getEnv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5"),
    parameters: ChatModelParameters = {},
    client?: AnthropicClientSettings | AnthropicClient,
  ) {
    const model = AnthropicClient.ensure(client).instance.languageModel(modelId);
    super(model);
    Object.assign(this.parameters, parameters ?? {});
  }

  protected get rejectsForcedToolChoice(): boolean {
    return NO_FORCED_TOOL_CHOICE_PREFIXES.some((prefix) => this.modelId.startsWith(prefix));
  }

  protected async transformInput(
    input: ChatModelInput,
  ): Promise<Parameters<typeof generateText<Record<string, any>>>[0]> {
    let transformed = await super.transformInput(input);
    // Anthropic's AI SDK adapter ignores topP when temperature is set.
    // Override the framework default, but preserve a temperature chosen by the caller.
    if (transformed.topP !== undefined && transformed.temperature === undefined) {
      transformed = { ...transformed, temperature: undefined };
    }
    // The tool call is requested through the response format, so don't offer the tools as well:
    // the model could call one directly, and the structured output would then be empty.
    if (this.rejectsForcedToolChoice && input.responseFormat) {
      transformed = { ...transformed, tools: undefined, toolChoice: undefined };
    }
    return transformed;
  }

  static {
    this.register();
  }
}
