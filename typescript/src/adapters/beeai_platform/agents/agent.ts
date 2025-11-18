/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter, EventMeta } from "@/emitter/emitter.js";
import { BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { ACPAgentUpdateEvent, ACPAgentErrorEvent } from "@/adapters/acp/agents/events.js";
import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import {
  BeeAIPlatformAgentInput,
  BeeAIPlatformAgentRunInput,
  BeeAIPlatformAgentRunOutput,
} from "./types.js";
import { BeeAIPlatformAgentEvents } from "./events.js";
import { ACPAgent } from "@/adapters/acp/agents/agent.js";
import { toCamelCase } from "remeda";

export class BeeAIPlatformAgent extends BaseAgent<
  BeeAIPlatformAgentRunInput,
  BeeAIPlatformAgentRunOutput
> {
  public readonly emitter: Emitter<BeeAIPlatformAgentEvents>;
  protected agent: ACPAgent;

  constructor(protected readonly input: BeeAIPlatformAgentInput) {
    super();
    this.agent = new ACPAgent(input);
    this.emitter = Emitter.root.child<BeeAIPlatformAgentEvents>({
      namespace: ["agent", "beeAIPlatform", toCamelCase(this.input.agentName)],
      creator: this,
    });
  }

  protected async _run(
    input: BeeAIPlatformAgentRunInput,
    _options: BaseAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<BeeAIPlatformAgentRunOutput> {
    const response = await this.agent.run(input).observe((emitter) => {
      emitter.on(
        "update",
        async (data: ACPAgentUpdateEvent, _: EventMeta) =>
          await context.emitter.emit("update", data),
      );
      emitter.on(
        "error",
        async (data: ACPAgentErrorEvent, _: EventMeta) => await context.emitter.emit("error", data),
      );
    });

    return { result: response.result, event: response.event };
  }

  async checkAgentExists() {
    return this.agent.checkAgentExists();
  }

  get memory() {
    return this.agent.memory;
  }

  set memory(memory: BaseMemory) {
    this.agent.memory = memory;
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      agent: this.agent.createSnapshot(),
      emitter: this.emitter,
    };
  }
}
