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

import { Callback } from "@/emitter/types.js";
import { Emitter } from "@/emitter/emitter.js";
import { AgentError, BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { Client as MCPClient } from "@i-am-bee/acp-sdk/client/index.js";
import { Transport } from "@i-am-bee/acp-sdk/shared/transport.js";
import { shallowCopy } from "@/serializer/utils.js";
import { NotImplementedError } from "@/errors.js";
import { AgentRunProgressNotificationSchema } from "@i-am-bee/acp-sdk/types.js";

export interface RemoteAgentRunInput {
  prompt: any;
}

export interface RemoteAgentRunOutput {
  message: Message;
}

export interface RemoteAgentEvents {
  update: Callback<{ output: string }>;
}

interface Input {
  client: MCPClient;
  transport: Transport;
  agent: string;
}

export class RemoteAgent extends BaseAgent<RemoteAgentRunInput, RemoteAgentRunOutput> {
  public emitter = Emitter.root.child<RemoteAgentEvents>({
    namespace: ["agent", "remote"],
    creator: this,
  });

  constructor(protected readonly input: Input) {
    super();
  }

  protected async _run(
    input: RemoteAgentRunInput,
    _options: BaseAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<RemoteAgentRunOutput> {
    if (input.prompt !== null) {
      await this.memory.add(new UserMessage(JSON.stringify(input.prompt, null, 2)));
    }

    const runner = await this.createRunner(context);

    const output = await runner(input);

    const finalMessage: Message = new AssistantMessage(output);

    await this.memory.add(finalMessage);

    return {
      message: finalMessage,
    };
  }

  protected async listAgents() {
    const response = await this.input.client.listAgents();

    return response.agents;
  }

  protected async createRunner(context: GetRunContext<this>) {
    const run = async (input: RemoteAgentRunInput): Promise<string> => {
      try {
        this.input.client.setNotificationHandler(
          AgentRunProgressNotificationSchema,
          async (notification) => {
            await context.emitter.emit("update", {
              output: JSON.stringify(notification.params.delta, null, 2),
            });
          },
        );
        if (!this.input.client.transport) {
          await this.input.client.connect(this.input.transport);
        }
      } catch (e) {
        throw new AgentError(`Can't connect to Beeai Platform.`, [e], { isFatal: true });
      }

      const agents = await this.listAgents();
      const agent = agents.find((agent) => agent.name === this.input.agent);
      if (!agent) {
        throw new AgentError(`Agent ${this.input.agent} is not registered in the platform`, [], {
          isFatal: true,
        });
      }

      const response = await this.input.client.runAgent(
        {
          name: this.input.agent,
          input: input.prompt,
        },
        {
          timeout: 10_000_000,
          signal: context.signal,
          onprogress: () => null,
        },
      );

      await this.input.client.close();
      const output = JSON.stringify(response.output, null, 2);
      await context.emitter.emit("update", { output });
      return output;
    };

    return run;
  }

  get memory() {
    return new UnconstrainedMemory();
  }

  set memory(memory: BaseMemory) {
    throw new NotImplementedError();
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      emitter: this.emitter,
    };
  }
}
