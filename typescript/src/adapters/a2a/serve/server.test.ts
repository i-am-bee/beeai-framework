/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer } from "node:http";
import { AddressInfo } from "node:net";
import express from "express";
import { A2AServer, A2AServerConfig } from "@/adapters/a2a/serve/server.js";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";

class FinalAnswerModel extends ChatModel {
  modelId = "final-answer";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "a2aServerTest"], creator: this });

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

async function serve(
  config: Partial<A2AServerConfig> = {},
  healthCheck?: () => Promise<boolean>,
): Promise<{ url: string; close: () => Promise<void> }> {
  const agent = new ToolCallingAgent({
    llm: new FinalAnswerModel(),
    tools: [],
    memory: new UnconstrainedMemory(),
  });
  const server = new A2AServer(Object.assign(new A2AServerConfig(), config));

  // listen first, so that the card advertises the port the test server got
  const outer = express();
  const httpServer = createServer(outer);
  await new Promise<void>((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(httpServer.address() as AddressInfo).port}`;

  server.register(agent, { url, healthCheck });
  outer.use((await server.createApp()).app);

  return {
    url,
    close: () =>
      new Promise((resolve) => {
        httpServer.closeAllConnections();
        httpServer.close(() => resolve());
      }),
  };
}

function legacyMessageSend(url: string) {
  // a 0.3 request: `message/send` with `kind` and lowercase roles, without the A2A-Version header
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "message/send",
      params: {
        message: {
          kind: "message",
          messageId: "msg-1",
          role: "user",
          parts: [{ kind: "text", text: "hello" }],
        },
      },
    }),
  }).then((res) => res.json());
}

describe("A2AServer", () => {
  it("lists the 1.0 and 0.3 interfaces in the agent card", async () => {
    const { url, close } = await serve();
    try {
      const res = await fetch(`${url}/.well-known/agent-card.json`, {
        headers: { "A2A-Version": "1.0" },
      });
      const card = await res.json();
      expect(card.supportedInterfaces).toEqual([
        { url, protocolBinding: "JSONRPC", tenant: "", protocolVersion: "1.0" },
        { url, protocolBinding: "JSONRPC", tenant: "", protocolVersion: "0.3" },
      ]);
    } finally {
      await close();
    }
  });

  it("lists only the 1.0 interface without enableV03Compat", async () => {
    const { url, close } = await serve({ enableV03Compat: false });
    try {
      const card = await fetch(`${url}/.well-known/agent-card.json`).then((res) => res.json());
      expect(card.supportedInterfaces).toEqual([
        { url, protocolBinding: "JSONRPC", tenant: "", protocolVersion: "1.0" },
      ]);
    } finally {
      await close();
    }
  });

  it("serves 0.3 requests with enableV03Compat", async () => {
    const { url, close } = await serve();
    try {
      const response = await legacyMessageSend(url);
      expect(response.error).toBeUndefined();
      expect(response.result.kind).toBe("task");
      expect(response.result.status.state).toBe("completed");
      expect(response.result.status.message.parts).toEqual([{ kind: "text", text: "done" }]);
    } finally {
      await close();
    }
  });

  it("rejects 0.3 requests without enableV03Compat", async () => {
    const { url, close } = await serve({ enableV03Compat: false });
    try {
      // a request without the A2A-Version header is taken as 0.3, which is not supported then
      const response = await legacyMessageSend(url);
      expect(response.error.code).toBe(-32009);
    } finally {
      await close();
    }
  });

  it("reports the health check", async () => {
    const healthy = await serve({}, async () => true);
    const notReady = await serve();
    try {
      const ok = await fetch(`${healthy.url}/health`);
      expect(ok.status).toBe(200);
      expect(await ok.text()).toBe("ok");

      // without a health check the server is ready only once serve() is listening
      const unavailable = await fetch(`${notReady.url}/health`);
      expect(unavailable.status).toBe(503);
    } finally {
      await healthy.close();
      await notReady.close();
    }
  });
});
