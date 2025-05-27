/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import { Emitter, EventMeta } from "@/emitter/emitter.js";
import { BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { ACPAgentUpdateEvent, ACPAgentErrorEvent } from "@/adapters/acp/agents/events.js";
import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { BeeAIPlatformAgentRunInput, BeeAIPlatformAgentRunOutput } from "./types.js";
import { BeeAIPlatformAgentEvents } from "./events.js";
import { ACPAgent } from "@/adapters/acp/agents/agent.js";

interface Input {
  url: string;
  agentName: string;
  memory: BaseMemory;
}

export class BeeAIPlatformAgent extends BaseAgent<
  BeeAIPlatformAgentRunInput,
  BeeAIPlatformAgentRunOutput
> {
  public emitter = Emitter.root.child<BeeAIPlatformAgentEvents>({
    namespace: ["agent", "beeai_platform"],
    creator: this,
  });
  protected agent: ACPAgent;

  constructor(protected readonly input: Input) {
    super();
    this.agent = new ACPAgent(input);
  }

  protected async _run(
    input: BeeAIPlatformAgentRunInput,
    _options: BaseAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<BeeAIPlatformAgentRunOutput> {
    async function updateEvent(data: ACPAgentUpdateEvent, _: EventMeta) {
      await context.emitter.emit("update", data);
    }

    async function errorEvent(data: ACPAgentErrorEvent, _: EventMeta) {
      await context.emitter.emit("error", data);
    }

    const response = await this.agent.run(input).observe((emitter) => {
      emitter.on("update", updateEvent);
      emitter.on("error", errorEvent);
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
