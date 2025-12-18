/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from "zod";
import { BaseToolRunOptions, StringToolOutput, Tool, ToolEmitter, ToolInput } from "./base.js";
import { BaseMemory } from "@/memory/base.js";
import { AssistantMessage, Message, SystemMessage, UserMessage } from "@/backend/message.js";
import { Emitter } from "@/emitter/emitter.js";
import { AnyAgent } from "@/agents/types.js";
import { GetRunContext } from "@/context.js";
import { getProp } from "@/internals/helpers/object.js";
import { toCamelCase } from "remeda";

export interface HandoffToolInput {
  name?: string;
  description?: string;
  propagateInputs?: boolean;
}

export class HandoffTool extends Tool<StringToolOutput> {
  private readonly propagateInputs: boolean;
  public readonly name: string;
  public readonly description: string;

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
    const memory = getProp(run.context, ["state", Tool.contextKeys.Memory]) as BaseMemory;

    let messages: Message[] = memory
      ? memory.messages.filter((msg) => !(msg instanceof SystemMessage))
      : [];

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (!(message instanceof AssistantMessage) || !message.getToolCalls()?.length) {
        messages = messages.slice(0, i + 1);
      } else {
        break;
      }
    }

    if (this.propagateInputs) {
      messages.push(new UserMessage(input.task));
    }

    this.target.memory.reset();

    const response = await this.target.run(messages);

    return new StringToolOutput(response.result.text);
  }
}
