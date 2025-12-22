/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";
import {
  BaseToolRunOptions,
  StringToolOutput,
  Tool,
  ToolEmitter,
  ToolError,
  ToolInput,
} from "./base.js";
import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage, Message, SystemMessage, UserMessage } from "@/backend/message.js";
import { Emitter } from "@/emitter/emitter.js";
import { AnyAgent } from "@/agents/types.js";
import { GetRunContext } from "@/context.js";
import { getProp } from "@/internals/helpers/object.js";
import { findLastIndex, toCamelCase } from "remeda";

export interface HandoffToolInput {
  name?: string;
  description?: string;
  propagateInputs?: boolean;
}

export class HandoffTool extends Tool<StringToolOutput> {
  private readonly propagateInputs: boolean;
  public name: string;
  public description: string;

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "handoff"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      task: z
        .string()
        .describe("Clearly defined task for the agent to work on based on his abilities."),
    });
  }

  constructor(
    private target: AnyAgent,
    options?: HandoffToolInput,
  ) {
    super();
    this.name = toCamelCase(options?.name || this.target.meta.name);
    this.description = options?.description || this.target.meta.description;
    this.propagateInputs = options?.propagateInputs ?? true;
  }

  protected async _run(
    input: ToolInput<this>,
    options: Partial<BaseToolRunOptions>,
    run: GetRunContext<typeof this>,
  ): Promise<StringToolOutput> {
    const memory = getProp(run.context, [Tool.contextKeys.Memory]) as BaseMemory;

    if (!memory) {
      throw new ToolError(`No memory found in the context.`, [], {
        isRetryable: false,
        isFatal: true,
      });
    }

    let messages: Message[] = memory.messages.filter((msg) => !(msg instanceof SystemMessage));

    const lastValidMsgIndex = findLastIndex(
      messages,
      (msg) => msg instanceof AssistantMessage && (msg.getToolCalls()?.length ?? 0) > 0,
    );

    if (lastValidMsgIndex !== -1) {
      messages = messages.slice(0, lastValidMsgIndex);
    }

    if (this.propagateInputs) {
      messages.push(new UserMessage(input.task));
    }

    const target = await this.target.clone();
    target.memory.reset();
    await target.memory.addMany(messages);

    const response = await target.run({ prompt: null });

    return new StringToolOutput(response.result.text);
  }
}
