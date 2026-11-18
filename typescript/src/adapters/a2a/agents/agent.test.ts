/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer } from "node:http";
import { AddressInfo } from "node:net";
import express, { Express } from "express";
import {
  AGENT_CARD_PATH,
  AgentCard,
  Artifact,
  Part,
  Role,
  StreamResponse,
  Task,
  TaskState,
} from "@a2a-js/sdk";
import {
  AgentEvent,
  AgentExecutor,
  DefaultRequestHandler,
  ExecutionEventBus,
  InMemoryTaskStore,
  RequestContext,
} from "@a2a-js/sdk/server";
import { agentCardHandler, jsonRpcHandler, UserBuilder } from "@a2a-js/sdk/server/express";
import { A2AAgent } from "@/adapters/a2a/agents/agent.js";
import { createA2AMessage, createA2APart } from "@/adapters/a2a/agents/utils.js";
import { A2AServer } from "@/adapters/a2a/serve/server.js";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, UserMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";

class FinalAnswerModel extends ChatModel {
  modelId = "final-answer";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "a2aAgentTest"], creator: this });

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  async _create(_input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
    return new ChatModelOutput([
      new AssistantMessage([
        {
          type: "tool-call",
          toolName: "final_answer",
          toolCallId: "call-1",
          input: { response: "done" },
        },
      ]),
    ]);
  }

  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}

// listen first, so that the card advertises the port the test server got
async function listen(build: (url: string) => Promise<Express>) {
  const outer = express();
  const httpServer = createServer(outer);
  await new Promise<void>((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(httpServer.address() as AddressInfo).port}`;
  outer.use(await build(url));
  return {
    url,
    close: () =>
      new Promise<void>((resolve) => {
        httpServer.closeAllConnections();
        httpServer.close(() => resolve());
      }),
  };
}

function createCard(url: string): AgentCard {
  return {
    name: "test",
    description: "test agent",
    supportedInterfaces: [{ url, protocolBinding: "JSONRPC", tenant: "", protocolVersion: "1.0" }],
    provider: undefined,
    version: "1.0.0",
    capabilities: { streaming: true, pushNotifications: false, extensions: [] },
    securitySchemes: {},
    securityRequirements: [],
    defaultInputModes: ["text"],
    defaultOutputModes: ["text"],
    skills: [],
    signatures: [],
  };
}

// an a2a-js server around a custom executor, to script the events A2AAgent receives
async function serveExecutor(execute: AgentExecutor["execute"]) {
  let agentCard: AgentCard | undefined;
  const server = await listen(async (url) => {
    agentCard = createCard(url);
    const requestHandler = new DefaultRequestHandler(agentCard, new InMemoryTaskStore(), {
      execute,
      cancelTask: async () => {},
    });
    const app = express();
    app.use(`/${AGENT_CARD_PATH}`, agentCardHandler({ agentCardProvider: requestHandler }));
    app.use("/", jsonRpcHandler({ requestHandler, userBuilder: UserBuilder.noAuthentication }));
    return app;
  });
  return { ...server, agentCard: agentCard! };
}

function createTask(context: RequestContext): Task {
  return (
    context.task ?? {
      id: context.taskId,
      contextId: context.contextId,
      status: { state: TaskState.TASK_STATE_SUBMITTED, message: undefined, timestamp: undefined },
      artifacts: [],
      history: [context.userMessage],
      metadata: undefined,
    }
  );
}

function publishStatus(
  bus: ExecutionEventBus,
  context: RequestContext,
  state: TaskState,
  text?: string,
) {
  const message = text
    ? createA2AMessage({
        role: Role.ROLE_AGENT,
        parts: [createA2APart({ $case: "text", value: text })],
        taskId: context.taskId,
        contextId: context.contextId,
      })
    : undefined;
  bus.publish(
    AgentEvent.statusUpdate({
      taskId: context.taskId,
      contextId: context.contextId,
      status: { state, message, timestamp: undefined },
      metadata: undefined,
    }),
  );
}

function publishArtifact(
  bus: ExecutionEventBus,
  context: RequestContext,
  text: string,
  append: boolean,
) {
  const artifact: Artifact = {
    artifactId: "answer",
    name: "",
    description: "",
    parts: [createA2APart({ $case: "text", value: text })],
    metadata: undefined,
    extensions: [],
  };
  bus.publish(
    AgentEvent.artifactUpdate({
      taskId: context.taskId,
      contextId: context.contextId,
      artifact,
      append,
      lastChunk: append,
      metadata: undefined,
    }),
  );
}

async function runCollecting(agent: A2AAgent, input: string) {
  const events: StreamResponse[] = [];
  const output = await agent.run({ input }).observe((emitter) => {
    emitter.on("update", (data) => {
      events.push(data.value);
    });
  });
  return { output, events };
}

function contextIdOf(events: StreamResponse[]): string | undefined {
  const payload = events.find((event) => event.payload?.$case === "task")?.payload;
  return payload?.$case === "task" ? payload.value.contextId : undefined;
}

function taskIdOf(events: StreamResponse[]): string | undefined {
  const payload = events.find((event) => event.payload?.$case === "task")?.payload;
  return payload?.$case === "task" ? payload.value.id : undefined;
}

describe("A2AAgent", () => {
  it("runs an agent served by A2AServer and reuses the context id across turns", async () => {
    const server = await listen(async (url) => {
      const a2aServer = new A2AServer();
      a2aServer.register(
        new ToolCallingAgent({
          llm: new FinalAnswerModel(),
          tools: [],
          memory: new UnconstrainedMemory(),
        }),
        { url },
      );
      return (await a2aServer.createApp()).app;
    });
    try {
      const agent = new A2AAgent({ url: server.url, memory: new UnconstrainedMemory() });

      const first = await runCollecting(agent, "hello");
      expect(first.output.result.text).toBe("done");
      expect(agent.memory.messages[0].text).toBe("hello");
      expect(agent.memory.messages.at(-1)?.text).toBe("done");

      // the first task is completed, so the second turn starts a new task in the same context
      const second = await runCollecting(agent, "again");
      expect(second.output.result.text).toBe("done");
      expect(contextIdOf(second.events)).toBe(contextIdOf(first.events));
      expect(taskIdOf(second.events)).not.toBe(taskIdOf(first.events));
    } finally {
      await server.close();
    }
  });

  it("returns a message reply", async () => {
    const server = await serveExecutor(async (context, bus) => {
      bus.publish(
        AgentEvent.message(
          createA2AMessage({
            role: Role.ROLE_AGENT,
            parts: [createA2APart({ $case: "text", value: "pong" })],
            contextId: context.contextId,
          }),
        ),
      );
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      const { output } = await runCollecting(agent, "ping");
      expect(output.result.text).toBe("pong");
      expect(output.event.payload?.$case).toBe("message");
    } finally {
      await server.close();
    }
  });

  it("keeps the task when the agent needs more input", async () => {
    const taskIds: string[] = [];
    const server = await serveExecutor(async (context, bus) => {
      taskIds.push(context.taskId);
      bus.publish(AgentEvent.task(createTask(context)));
      if (taskIds.length === 1) {
        publishStatus(bus, context, TaskState.TASK_STATE_INPUT_REQUIRED, "which city?");
      } else {
        publishStatus(bus, context, TaskState.TASK_STATE_COMPLETED, "sunny");
      }
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      expect((await runCollecting(agent, "weather?")).output.result.text).toBe("which city?");
      expect((await runCollecting(agent, "Prague")).output.result.text).toBe("sunny");
      expect(taskIds[1]).toBe(taskIds[0]);
      // the follow-up turn gets the whole task again, the earlier reply is not added twice
      expect(agent.memory.messages.map((m) => [m.role, m.text])).toEqual([
        ["user", "weather?"],
        ["assistant", "which city?"],
        ["user", "Prague"],
        ["assistant", "sunny"],
      ]);
    } finally {
      await server.close();
    }
  });

  it("drops a pending task when the run switches to another context", async () => {
    const received: { taskId?: string; contextId?: string }[] = [];
    const server = await serveExecutor(async (context, bus) => {
      received.push({ taskId: context.taskId, contextId: context.contextId });
      bus.publish(AgentEvent.task(createTask(context)));
      publishStatus(
        bus,
        context,
        received.length === 1
          ? TaskState.TASK_STATE_INPUT_REQUIRED
          : TaskState.TASK_STATE_COMPLETED,
        received.length === 1 ? "which city?" : "done",
      );
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      expect((await runCollecting(agent, "weather?")).output.result.text).toBe("which city?");
      const output = await agent.run({ input: "hello" }, { contextId: "other-context" });
      expect(output.result.text).toBe("done");
      expect(received[1].contextId).toBe("other-context");
      expect(received[1].taskId).not.toBe(received[0].taskId);
    } finally {
      await server.close();
    }
  });

  it("does not keep the task id of a message reply that stores no task", async () => {
    const server = await serveExecutor(async (context, bus) => {
      bus.publish(
        AgentEvent.message(
          createA2AMessage({
            role: Role.ROLE_AGENT,
            parts: [createA2APart({ $case: "text", value: "pong" })],
            contextId: context.contextId,
            taskId: context.taskId,
          }),
        ),
      );
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      await runCollecting(agent, "ping");
      // resending the id of a task that was never stored would fail with TaskNotFoundError
      const { output } = await runCollecting(agent, "ping again");
      expect(output.result.text).toBe("pong");
    } finally {
      await server.close();
    }
  });

  it("sends file data as raw bytes and URLs as url parts", async () => {
    const received: Part["content"][] = [];
    const server = await serveExecutor(async (context, bus) => {
      received.push(...context.userMessage.parts.map((part) => part.content));
      bus.publish(
        AgentEvent.message(
          createA2AMessage({
            role: Role.ROLE_AGENT,
            parts: [createA2APart({ $case: "text", value: "got it" })],
            contextId: context.contextId,
          }),
        ),
      );
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      await agent.run({
        input: new UserMessage([
          { type: "file", data: Buffer.from("png").toString("base64"), mediaType: "image/png" },
          { type: "file", data: "https://example.com/logo.png", mediaType: "image/png" },
        ]),
      });
      expect(received[0]?.$case).toBe("raw");
      expect(Buffer.from(received[0]?.value as Uint8Array).toString()).toBe("png");
      expect(received[1]).toEqual({ $case: "url", value: "https://example.com/logo.png" });
    } finally {
      await server.close();
    }
  });

  it("rebuilds an artifact streamed in chunks", async () => {
    const server = await serveExecutor(async (context, bus) => {
      bus.publish(AgentEvent.task(createTask(context)));
      publishArtifact(bus, context, "Hel", false);
      publishArtifact(bus, context, "lo", true);
      publishStatus(bus, context, TaskState.TASK_STATE_COMPLETED);
      bus.finished();
    });
    try {
      const agent = new A2AAgent({
        agentCard: server.agentCard,
        memory: new UnconstrainedMemory(),
      });
      const { output } = await runCollecting(agent, "greet");
      expect(output.result.text).toBe("Hello");
    } finally {
      await server.close();
    }
  });
});
