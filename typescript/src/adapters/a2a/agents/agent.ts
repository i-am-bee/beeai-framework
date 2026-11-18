/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

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
  TaskState,
  Artifact,
  Part,
  Role,
  StreamResponse,
} from "@a2a-js/sdk";
import {
  ClientFactory,
  ClientFactoryOptions,
  DefaultAgentCardResolver,
  JsonRpcTransportFactory,
  RestTransportFactory,
} from "@a2a-js/sdk/client";
import { Message } from "@/backend/message.js";
import { convertA2AMessageToFrameworkMessage, createA2AMessage, createA2APart } from "./utils.js";

const INTERRUPTED_TASK_STATES = new Set([
  TaskState.TASK_STATE_INPUT_REQUIRED,
  TaskState.TASK_STATE_AUTH_REQUIRED,
]);

// a2a-js talks to agents that still serve 0.3 only when its compat layer is enabled
const legacyCompat = { enabled: true };
const cardResolver = new DefaultAgentCardResolver({ legacyCompat });
const clientFactory = new ClientFactory(
  ClientFactoryOptions.createFrom(ClientFactoryOptions.default, {
    cardResolver,
    transports: [
      new JsonRpcTransportFactory({ legacyCompat }),
      new RestTransportFactory({ legacyCompat }),
    ],
  }),
);

export class A2AAgent extends BaseAgent<A2AAgentRunInput, A2AAgentRunOutput, A2AAgentRunOptions> {
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
    return this.agentCard ? this.agentCard.name : `agent_${this.url?.split(":").at(-1)}`;
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

    let lastEvent: StreamResponse | null = null;
    const messages: A2AMessage[] = [];
    let task: Task | null = null;

    try {
      const client = await clientFactory.createFromAgentCard(this.agentCard!);
      const message = this.convertToA2AMessage(input.input);
      const stream = client.sendMessageStream({
        tenant: "",
        message,
        configuration: undefined,
        metadata: undefined,
      });

      for await (const event of stream) {
        lastEvent = event;
        const payload = event.payload;
        if (!payload) {
          continue;
        }

        let taskId: string, contextId: string;
        if (payload.$case === "message") {
          ({ taskId, contextId } = payload.value);
          messages.push(payload.value);
        } else {
          task = mergeTaskEvent(task, payload);
          ({ id: taskId, contextId } = task);
        }

        if (taskId && taskId !== this.taskId) {
          this.taskId = taskId;
        }
        if (contextId && contextId !== this.contextId) {
          this.contextId = contextId;
        }
        if (taskId && !this.referenceTaskIds.includes(taskId)) {
          this.referenceTaskIds.push(taskId);
        }

        await context.emitter.emit("update", { value: event });
      }

      if (!lastEvent) {
        throw new AgentError("No result received from agent.");
      }

      // only a task that waits for input is continued by the next turn; a message reply may carry
      // a task id without a stored task, and a terminal task can't receive messages anymore
      this.taskId =
        task?.status && INTERRUPTED_TASK_STATES.has(task.status.state) ? task.id : undefined;

      // add input messages to memory
      const inputMessages = Array.isArray(input.input) ? input.input : [input.input];
      await this.memory.addMany(inputMessages.map(this.convertToFrameworkMessage));

      let results: (A2AMessage | Artifact)[];
      if (task) {
        // a follow-up turn gets the whole task again, the earlier replies are already in memory
        const history = historyWithStatus(task);
        const sentIndex = history.findIndex((m) => m.messageId === message.messageId);
        const agentHistory = history.slice(sentIndex + 1).filter((m) => m.role === Role.ROLE_AGENT);
        if (task.artifacts.length > 0) {
          results = task.artifacts;
        } else if (agentHistory.length > 0) {
          results = agentHistory;
        } else if (task.status?.message) {
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
      // the task may be gone on the server, resending its id would fail every following run
      this.taskId = undefined;
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
    // a task belongs to its context, so a pending task is dropped when the context changes
    if (contextId && contextId !== this.contextId && !taskId) {
      this.taskId = undefined;
    }
    if (contextId) {
      this.contextId = contextId;
    }
    // keep the task of a previous turn (e.g. one that requires more input) unless another is given
    this.taskId = taskId || this.taskId;
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
      // the full card URL with an empty path, as the card path may be relative to the agent URL
      this.agentCard = await cardResolver.resolve(new URL(this.agentCardPath, this.url).href, "");
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
      return createA2AMessage({
        role: Role.ROLE_USER,
        parts: [createA2APart({ $case: "text", value: input })],
        contextId: this.contextId,
        taskId: this.taskId,
        referenceTaskIds: this.referenceTaskIds,
      });
    } else if (input instanceof Message) {
      return createA2AMessage({
        role: input.role == "assistant" ? Role.ROLE_AGENT : Role.ROLE_USER,
        parts: input.content.map(convertFrameworkContentToA2APart),
        contextId: this.contextId,
        taskId: this.taskId,
        referenceTaskIds: this.referenceTaskIds,
      });
    } else if (!Array.isArray(input) && "messageId" in input) {
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
    } else if ("messageId" in input || "artifactId" in input) {
      return convertA2AMessageToFrameworkMessage(input);
    } else {
      throw new AgentError("Unsupported input type");
    }
  }
}

type TaskEventPayload = Extract<
  NonNullable<StreamResponse["payload"]>,
  { $case: "task" | "statusUpdate" | "artifactUpdate" }
>;

// a2a-js 1.x streams the task events as they come, the task is rebuilt as its server-side ResultManager does
function mergeTaskEvent(task: Task | null, payload: TaskEventPayload): Task {
  if (payload.$case === "task") {
    return structuredClone(payload.value);
  }

  const update = payload.value;
  const merged: Task = task ?? {
    id: update.taskId,
    contextId: update.contextId,
    status: undefined,
    artifacts: [],
    history: [],
    metadata: undefined,
  };
  if (update.metadata) {
    merged.metadata = { ...merged.metadata, ...update.metadata };
  }

  if (payload.$case === "statusUpdate") {
    merged.status = payload.value.status;
    const message = payload.value.status?.message;
    if (message && !merged.history.some((m) => m.messageId === message.messageId)) {
      merged.history.push(message);
    }
  } else if (payload.value.artifact) {
    const { artifact, append } = payload.value;
    const index = merged.artifacts.findIndex((a) => a.artifactId === artifact.artifactId);
    if (index === -1) {
      merged.artifacts.push(artifact);
    } else if (append) {
      const existing = merged.artifacts[index];
      merged.artifacts[index] = {
        ...existing,
        parts: [...existing.parts, ...artifact.parts],
        name: artifact.name || existing.name,
        description: artifact.description || existing.description,
        metadata: artifact.metadata
          ? { ...existing.metadata, ...artifact.metadata }
          : existing.metadata,
      };
    } else {
      merged.artifacts[index] = artifact;
    }
  }
  return merged;
}

// a task received whole (e.g. from a Python server) can still hold its last message only in its status
function historyWithStatus(task: Task): A2AMessage[] {
  const message = task.status?.message;
  if (message && !task.history.some((m) => m.messageId === message.messageId)) {
    return [...task.history, message];
  }
  return task.history;
}

function convertFrameworkContentToA2APart(content: MessageContentPart): Part {
  const { type, ...rest } = content;
  switch (type) {
    case "text":
      return createA2APart({ $case: "text", value: content.text });
    case "file":
    case "image": {
      const data = type === "file" ? content.data : content.image;
      const options = {
        mediaType: content.mediaType,
        filename: "filename" in content ? content.filename : undefined,
        metadata: { providerOptions: content.providerOptions },
      };
      if (data instanceof URL) {
        return createA2APart({ $case: "url", value: data.href }, options);
      }
      // URLs kept as strings (e.g. an a2a `url` part converted to the framework) stay URLs
      if (typeof data === "string" && /^(https?|data):/i.test(data)) {
        return createA2APart({ $case: "url", value: data }, options);
      }
      // a2a-js 1.x carries the raw bytes, string file data in the framework is base64
      const raw =
        typeof data === "string"
          ? Buffer.from(data, "base64")
          : Buffer.from(data instanceof ArrayBuffer ? new Uint8Array(data) : data);
      return createA2APart({ $case: "raw", value: raw }, options);
    }
    case "tool-call":
    case "tool-result":
      return createA2APart({ $case: "data", value: rest });
  }
}
