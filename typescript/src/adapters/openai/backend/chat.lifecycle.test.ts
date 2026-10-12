/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer } from "node:http";
import { once } from "node:events";
import { AddressInfo } from "node:net";
import { OpenAIChatModel } from "@/adapters/openai/backend/chat.js";
import { UserMessage } from "@/backend/message.js";
import { FrameworkError } from "@/errors.js";
import { describe, expect, it } from "vitest";

describe("Chat model run cleanup", () => {
  it.each(["completed", "cancelled", "failed"])(
    "Does not retain %s streamed runs on a long-lived model",
    async (mode) => {
      let disconnected = 0;
      const server = createServer((request, response) => {
        request.resume();
        response.on("close", () => disconnected++);
        if (mode === "failed") {
          response.writeHead(400, { "content-type": "application/json" });
          response.end(JSON.stringify({ error: { message: "test provider failure" } }));
          return;
        }
        response.writeHead(200, { "content-type": "text/event-stream" });
        const chunk = (delta: object, finishReason: string | null) =>
          JSON.stringify({
            id: "test-stream",
            object: "chat.completion.chunk",
            created: 0,
            model: "test-model",
            choices: [{ index: 0, delta, finish_reason: finishReason }],
          });
        response.write(`data: ${chunk({ role: "assistant", content: "hello" }, null)}\n\n`);
        if (mode === "completed") {
          response.end(`data: ${chunk({}, "stop")}\n\ndata: [DONE]\n\n`);
        }
        // Cancelled streams stay open until the real client aborts the HTTP request.
      });
      server.listen(0, "127.0.0.1");
      await once(server, "listening");
      const model = new OpenAIChatModel(
        "test-model",
        {},
        {
          apiKey: "local-test-only",
          baseURL: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`,
        },
      );
      const baseline = model.emitter.createSnapshot().cleanups.length;
      try {
        for (let index = 0; index < 10; index++) {
          const controller = new AbortController();
          const reason = new Error("cancel streamed request");
          const run = model
            .create({
              messages: [new UserMessage("hi")],
              stream: true,
              maxRetries: 0,
              abortSignal: controller.signal,
            })
            .observe((emitter) => {
              if (mode === "cancelled") {
                emitter.on("newToken", () => controller.abort(reason));
              }
            });
          if (mode === "completed") {
            expect((await run).getTextContent()).toBe("hello");
          } else {
            await expect(run).rejects.toBeInstanceOf(FrameworkError);
          }
        }
        await vi.waitFor(() => expect(disconnected).toBe(10));
        expect(model.emitter.createSnapshot().cleanups).toHaveLength(baseline);
      } finally {
        model.destroy();
        const closed = once(server, "close");
        server.close();
        server.closeAllConnections();
        await closed;
      }
    },
    10000,
  );
});
