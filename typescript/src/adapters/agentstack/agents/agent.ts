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
  AgentStackAgentInput,
  AgentStackAgentRunInput,
  AgentStackAgentRunOutput,
} from "./types.js";
import { AgentStackAgentEvents } from "./events.js";
import { ACPAgent } from "@/adapters/acp/agents/agent.js";
import { toCamelCase } from "remeda";
import { Logger } from "@/logger/logger.js";

/**
 * @deprecated AgentStack is no longer maintained and the AgentStack integration will be removed in a future
 * release. Use A2AAgent (`beeai-framework/adapters/a2a/agents/agent`) for agents exposed over the A2A
 * protocol, or ACPAgent (`beeai-framework/adapters/acp/agents/agent`), which this class wraps.
 */
export class AgentStackAgent extends BaseAgent<AgentStackAgentRunInput, AgentStackAgentRunOutput> {
  public readonly emitter: Emitter<AgentStackAgentEvents>;
  protected agent: ACPAgent;

  constructor(protected readonly input: AgentStackAgentInput) {
    super();
    Logger.root.warn(
      "The AgentStackAgent is deprecated and will be removed soon. AgentStack is no longer maintained. Use A2AAgent (for agents exposed over the A2A protocol) or ACPAgent instead.",
    );
    this.agent = new ACPAgent(input);
    this.emitter = Emitter.root.child<AgentStackAgentEvents>({
      namespace: ["agent", "agentStack", toCamelCase(this.input.agentName)],
      creator: this,
    });
  }

  protected async _run(
    input: AgentStackAgentRunInput,
    _options: BaseAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<AgentStackAgentRunOutput> {
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
