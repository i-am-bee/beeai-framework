/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { v4 as uuid4 } from "uuid";
import { Emitter } from "@/emitter/emitter.js";
import { AgentError, BaseAgent } from "@/agents/base.js";
import { GetRunContext } from "@/context.js";
import { AssistantMessage, MessageContentPart, UserMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { A2AAgentInput, A2AAgentRunInput, A2AAgentRunOutput, A2AAgentRunOptions } from "./types.js";
import { A2AAgentEvents } from "./events.js";
import { toCamelCase } from "remeda";
import {
  AGENT_CARD_PATH,
  AgentCard,
  Message as A2AMessage,
  Task,
  Artifact,
  Part,
  FileWithUri,
  FileWithBytes,
} from "@a2a-js/sdk";
import { A2AClient } from "@a2a-js/sdk/client";
import { Message } from "@/backend/message.js";
import { convertA2AMessageToFrameworkMessage } from "./utils.js";

export class A2AAgent extends BaseAgent<A2AAgentRunInput, A2AAgentRunOutput> {
  private _memory!: BaseMemory;
  public readonly emitter: Emitter<A2AAgentEvents>;
  protected url: string | undefined;
  public agentCard: AgentCard | undefined;
  protected agentCardPath: string;
  protected contextId: string | undefined;
  protected taskId: string | undefined;
  protected referenceTaskIds: string[] = [];

  constructor(protected readonly input: A2AAgentInput) {
    super();
    this.emitter = Emitter.root.child<A2AAgentEvents>({
      namespace: ["a2a", "agent", toCamelCase(this.name)],
      creator: this,
    });
    if (input.url) {
      this.url = input.url;
    } else if (input.agentCard) {
      this.agentCard = input.agentCard;
    } else {
      throw new AgentError("Either url or agentCard must be provided in A2AAgentInput", [], {
        isFatal: true,
      });
    }

    this.memory = input.memory;
    this.agentCardPath = input.agentCardPath || AGENT_CARD_PATH;
  }

  get name() {
    return this.agentCard ? this.agentCard.name : `agent_${this.url?.split(":")[-1]}`;
  }

  protected async _run(
    input: A2AAgentRunInput,
    _options: A2AAgentRunOptions,
    context: GetRunContext<this>,
  ): Promise<A2AAgentRunOutput> {
    this.setRunParams(_options?.contextId, _options?.taskId, _options?.clearContext);

    if (!this.agentCard) {
      await this.loadAgentCard();
    }

    const client = new A2AClient(this.agentCard!);

    let lastEvent = null;
    const messages: (A2AMessage | Artifact)[] = [];
    let task: Task | null = null;

    try {
      const stream = client.sendMessageStream({ message: this.convertToA2AMessage(input.input) });

      for await (const event of stream) {
        lastEvent = event;

        const taskId = event.kind === "task" ? event.id : event.taskId;
        const contextId = event.contextId;
        if (taskId && taskId !== this.taskId) {
          this.taskId = taskId;
        }
        if (contextId && contextId !== this.contextId) {
          this.contextId = contextId;
        }
        if (taskId && !this.referenceTaskIds.includes(taskId)) {
          this.referenceTaskIds.push(taskId);
        }

        if (event.kind === "status-update" || event.kind === "artifact-update") {
          const content = event.kind === "status-update" ? event.status.message : event.artifact;
          if (content) {
            messages.push(content);
          }
          if (
            event.kind === "status-update" &&
            event.final &&
            event.status.state !== "input-required"
          ) {
            this.taskId = undefined;
          }
        } else if (event.kind === "message") {
          messages.push(event);
        } else if (event.kind === "task") {
          task = event;
        }

        await context.emitter.emit("update", { value: event });
      }

      if (!lastEvent) {
        throw new AgentError("No result received from agent.");
      }

      // add input messages to memory
      const inputMessages = Array.isArray(input.input) ? input.input : [input.input];
      await this.memory.addMany(inputMessages.map(this.convertToFrameworkMessage));

      let results: (A2AMessage | Artifact)[];
      if (task) {
        if (task.artifacts) {
          results = task.artifacts;
        } else if (task.history) {
          results = task.history;
        } else if (task.status.message) {
          results = [task.status.message];
        } else {
          results = [];
        }
      } else {
        results = messages;
      }

      // retrieve the assistant's response
      if (results.length > 0) {
        const assistantMessages = results.map(this.convertToFrameworkMessage);
        await this.memory.addMany(assistantMessages);
        return {
          result: assistantMessages[assistantMessages.length - 1],
          event: lastEvent,
        };
      } else {
        return {
          result: new AssistantMessage("No response from agent."),
          event: lastEvent,
        };
      }
    } catch (err) {
      let message: string;

      if (err.message) {
        message = err.message;
      } else if (err.error) {
        message = err.error.toString();
      } else {
        message = "Unknown error";
      }

      await context.emitter.emit("error", { message });

      throw new AgentError(message, [err], {
        context: lastEvent ?? undefined,
      });
    }
  }

  protected setRunParams(contextId?: string, taskId?: string, clearContext = false) {
    if (contextId) {
      this.contextId = contextId;
    }
    this.taskId = taskId;
    if (clearContext) {
      this.contextId = undefined;
      this.taskId = undefined;
      this.referenceTaskIds = [];
    }
  }

  async loadAgentCard() {
    if (this.agentCard) {
      return;
    }

    if (!this.url) {
      throw new AgentError("URL must be provided to load the agent card.", [], { isFatal: true });
    }

    try {
      const client = await A2AClient.fromCardUrl(new URL(this.agentCardPath, this.url).href, {
        agentCardPath: this.agentCardPath,
      });
      this.agentCard = await client.getAgentCard();
    } catch (error) {
      throw new AgentError(`Can not load agent card: ${error.message}`, [error], {
        isFatal: true,
      });
    }
  }

  async checkAgentExists() {
    await this.loadAgentCard();
    if (!this.agentCard) {
      throw new AgentError(`Agent ${this.name} does not exist.`, [], { isFatal: true });
    }
  }

  get memory() {
    return this._memory;
  }

  set memory(memory: BaseMemory) {
    if (!memory.isEmpty()) {
      throw new AgentError("Memory must be empty before setting.");
    }
    this._memory = memory;
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: shallowCopy(this.input),
      emitter: this.emitter,
    };
  }

  protected convertToA2AMessage(input: string | Message | Message[] | A2AMessage): A2AMessage {
    if (typeof input === "string") {
      return {
        kind: "message",
        role: "user",
        parts: [
          {
            kind: "text",
            text: input,
          },
        ],
        messageId: uuid4(),
        contextId: this.contextId,
        taskId: this.taskId,
        referenceTaskIds: this.referenceTaskIds,
      };
    } else if (input instanceof Message) {
      const parts = input.content.map(convertFrameworkContentoA2APart);
      return {
        kind: "message",
        role: input.role == "assistant" ? "agent" : "user",
        parts: parts,
        messageId: uuid4(),
        contextId: this.contextId,
        taskId: this.taskId,
        referenceTaskIds: this.referenceTaskIds,
      };
    } else if ("kind" in input && input.kind === "message") {
      return input;
    } else if (Array.isArray(input) && input.length > 0) {
      return this.convertToA2AMessage(input[input.length - 1]);
    } else {
      throw new AgentError("Unsupported input type");
    }
  }

  protected convertToFrameworkMessage(input: string | Message | A2AMessage | Artifact): Message {
    if (typeof input === "string") {
      return new UserMessage(input);
    } else if (input instanceof Message) {
      return input;
    } else if (("kind" in input && input.kind === "message") || "artifactId" in input) {
      return convertA2AMessageToFrameworkMessage(input);
    } else {
      throw new AgentError("Unsupported input type");
    }
  }
}

function convertFrameworkContentoA2APart(content: MessageContentPart): Part {
  const { type, ...rest } = content;
  switch (type) {
    case "text":
      return { kind: "text", text: content.text };
    case "file":
    case "image": {
      const data = type === "file" ? content.data : content.image;
      const decoder = new TextDecoder("utf-8");
      const file =
        data instanceof URL
          ? ({ uri: data.href } as FileWithUri)
          : ({
              bytes: typeof data === "string" ? data : decoder.decode(data),
            } as FileWithBytes);
      return {
        kind: "file",
        file: {
          ...file,
          mimeType: content.mimeType,
          name: "filename" in content ? content.filename : undefined,
        },
        metadata: {
          providerOptions: content.providerOptions,
        },
      };
    }
    case "tool-call":
    case "tool-result":
      return {
        kind: "data",
        data: rest,
      };
  }
}
