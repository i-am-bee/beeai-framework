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

import { Callback } from "@/emitter/types.js";
import { Emitter } from "@/emitter/emitter.js";
import { AgentError, BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";

export interface RemoteAgentRunInput {
  input: Message | string | Message[] | string[];
}

export interface RemoteAgentRunOutput {
  result: Message;
  event: Record<string, any>;
}

export interface RemoteAgentEvents {
  update: Callback<{ key: string; value: any }>;
  error: Callback<{ message: string }>;
}

interface Input {
  url: string;
  agentName: string;
  memory: BaseMemory;
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
    const inputs = Array.isArray(input.input)
      ? input.input.map(this.convertToACPMessage)
      : [this.convertToACPMessage(input.input)];

    const url = new URL(this.input.url);
    url.pathname += "/runs";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        agent_name: this.input.agentName,
        input: inputs,
        mode: "stream",
      }),
      signal: context.signal,
    });

    if (!response.ok) {
      throw new AgentError(`HTTP error! status: ${response.status}`, [], {
        context: { message: await response.text() },
      });
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new AgentError("Agent's response is not valid");
    }

    const decoder = new TextDecoder();
    let partialData = "";
    let eventData: any = null;

    function parseSSEEvent(eventString: string) {
      const lines = eventString.split("\n");
      const event: any = {};
      for (const line of lines) {
        const [key, ...valueParts] = line.split(":");
        if (key && valueParts && valueParts.length > 0) {
          const value = valueParts.join(":").trim();
          event[key.trim()] = value;
        }
      }
      return event;
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      partialData += decoder.decode(value, { stream: true });

      const events = partialData.split("\n\n");
      partialData = events.pop() || "";

      await Promise.all(
        events.map(async (eventString) => {
          if (eventString.trim() !== "") {
            const event = parseSSEEvent(eventString);
            if (event) {
              try {
                eventData = JSON.parse(event.data);
                await context.emitter.emit("update", {
                  key: eventData.type,
                  value: { ...eventData, type: undefined },
                });
              } catch {
                await context.emitter.emit("error", {
                  message: "Error parsing JSON:",
                });
              }
            }
          }
        }),
      );
    }

    if (!eventData) {
      throw new AgentError("No event received from agent.");
    }

    if (eventData.type === "run.failed") {
      const message =
        eventData.run?.error?.message || "Something went wrong with the agent communication.";
      await context.emitter.emit("error", { message });
      throw new AgentError(message);
    } else if (eventData.type === "run.completed") {
      const text = eventData.run.output.reduce(
        (acc: string, output: any) =>
          acc + output.parts.reduce((acc2: string, part: any) => acc2 + part.content, ""),
        "",
      );
      const assistantMessage: Message = new AssistantMessage(text, { event: eventData });
      const inputMessages = Array.isArray(input.input)
        ? input.input.map(this.convertToMessage)
        : [this.convertToMessage(input.input)];

      await this.memory.addMany(inputMessages);
      await this.memory.add(assistantMessage);

      return { result: assistantMessage, event: eventData };
    } else {
      return { result: new AssistantMessage("No response from agent."), event: eventData };
    }
  }

  async checkAgentExists() {
    const url = new URL(this.input.url);
    url.pathname += "/agents";
    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const agents = data.agents;
      const agent = agents.find((agent: any) => agent.name === this.input.agentName);
      return !!agent;
    } catch (error) {
      throw new AgentError(`Error while checking agent existence: ${error.message}`, [], {
        isFatal: true,
      });
    }
  }

  get memory() {
    return this.input.memory;
  }

  set memory(memory: BaseMemory) {
    this.input.memory = memory;
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      emitter: this.emitter,
    };
  }

  protected convertToACPMessage(input: string | Message): any {
    if (typeof input === "string") {
      return { parts: [{ content: input }] };
    } else if (input instanceof Message) {
      return { parts: [{ content: input.content }] };
    } else {
      throw new AgentError("Unsupported input type");
    }
  }

  protected convertToMessage(input: string | Message): any {
    if (typeof input === "string") {
      return new UserMessage(input);
    } else if (input instanceof Message) {
      return input;
    } else {
      throw new AgentError("Unsupported input type");
    }
  }
}
