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
import { AssistantMessage, Message } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { Client as MCPClient } from "@i-am-bee/acp-sdk/client/index.js";
import { Transport } from "@i-am-bee/acp-sdk/shared/transport.js";
import { shallowCopy } from "@/serializer/utils.js";
import { NotImplementedError } from "@/errors.js";
import { AgentRunProgressNotificationSchema } from "@i-am-bee/acp-sdk/types.js";
import { SSEClientTransport } from "@i-am-bee/acp-sdk/client/sse.js";

export interface RemoteAgentRunInput {
  prompt: Record<string, unknown> | string;
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
  agentName: string;
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
    const runner = this.createRunner(context);

    const output = await runner(input);

    const message: Message = new AssistantMessage(output);

    return { message };
  }

  protected async listAgents() {
    const response = await this.input.client.listAgents();

    return response.agents;
  }

  protected createRunner(context: GetRunContext<this>) {
    return async (input: RemoteAgentRunInput): Promise<string> => {
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
      const agent = agents.find((agent) => agent.name === this.input.agentName);
      if (!agent) {
        throw new AgentError(
          `Agent ${this.input.agentName} is not registered in the platform`,
          [],
          {
            isFatal: true,
          },
        );
      }

      const response = await this.input.client.runAgent(
        {
          name: this.input.agentName,
          input: typeof input.prompt === "string" ? { text: input.prompt } : input.prompt,
        },
        {
          timeout: 10_000_000,
          signal: context.signal,
          onprogress: () => null, // This has to be here in order for notifications to work.
        },
      );

      const output = JSON.stringify(response.output, null, 2);
      await context.emitter.emit("update", { output });
      return output;
    };
  }

  get memory() {
    throw new NotImplementedError();
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

  static createSSEAgent(url: string, agentName: string) {
    return new RemoteAgent({
      client: new MCPClient({
        name: "remote-agent",
        version: "1.0.0",
      }),
      transport: new SSEClientTransport(new URL(url)),
      agentName: agentName,
    });
  }
}
