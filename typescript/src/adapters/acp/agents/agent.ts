/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { AgentError, BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { AssistantMessage, Message, UserMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { RestfulClient } from "@/internals/fetcher.js";
import { ACPAgentInput, ACPAgentRunInput, ACPAgentRunOutput } from "./types.js";
import { ACPAgentEvents } from "./events.js";
import { toCamelCase } from "remeda";

export class ACPAgent extends BaseAgent<ACPAgentRunInput, ACPAgentRunOutput> {
  public readonly emitter: Emitter<ACPAgentEvents>;
  protected client: RestfulClient<{ runs: string; agents: string }>;

  constructor(protected readonly input: ACPAgentInput) {
    super();
    this.emitter = Emitter.root.child<ACPAgentEvents>({
      namespace: ["agent", "acp", toCamelCase(this.input.agentName)],
      creator: this,
    });
    this.client = new RestfulClient({
      baseUrl: this.input.url,
      headers: async () => ({
        "Accept": "application/json",
        "Content-Type": "application/json",
      }),
      paths: { runs: `/runs`, agents: `/agents` },
    });
  }

  protected async _run(
    input: ACPAgentRunInput,
    _options: BaseAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<ACPAgentRunOutput> {
    const inputs = Array.isArray(input.input)
      ? input.input.map(this.convertToACPMessage)
      : [this.convertToACPMessage(input.input)];

    const generator = this.client.stream("runs", {
      body: JSON.stringify({
        agent_name: this.input.agentName,
        input: inputs,
        mode: "stream",
      }),
      signal: context.signal,
    });

    let eventData: any = null;
    for await (const event of generator) {
      try {
        eventData = JSON.parse(event.data);
        await context.emitter.emit("update", {
          key: eventData.type,
          value: { ...eventData, type: undefined },
        });
      } catch {
        await context.emitter.emit("error", {
          message: "Error parsing JSON",
        });
      }
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
    try {
      const response = await this.client.fetch("agents");
      return !!response.agents.find((agent: any) => agent.name === this.input.agentName);
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
      return { parts: [{ content: input, role: "user" }] };
    } else if (input instanceof Message) {
      return { parts: [{ content: input.content, role: input.role }] };
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
