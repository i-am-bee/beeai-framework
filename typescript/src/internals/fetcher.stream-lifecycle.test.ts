/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { RestfulClient } from "@/internals/fetcher.js";

describe("RestfulClient stream lifecycle", () => {
  let server: Server;
  let baseUrl: string;
  let disconnected: boolean;

  beforeEach(async () => {
    disconnected = false;
    server = createServer((request, response) => {
      request.on("close", () => {
        disconnected = true;
      });
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.write("data: first\n\n");
      if (request.url === "/complete") {
        response.end();
      }
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it("closes the HTTP stream when the consumer returns early", async () => {
    const client = new RestfulClient({ baseUrl, paths: { events: "/events" } });
    const stream = client.stream("events", { method: "GET" });
    expect((await stream.next()).value).toMatchObject({ data: "first" });

    await stream.return();
    await expect.poll(() => disconnected, { timeout: 1000 }).toBe(true);
  });

  it("preserves cancellation from the caller's signal", async () => {
    const client = new RestfulClient({ baseUrl, paths: { events: "/events" } });
    const controller = new AbortController();
    const stream = client.stream("events", { method: "GET", signal: controller.signal });
    expect((await stream.next()).value).toMatchObject({ data: "first" });

    controller.abort();
    await expect(stream.next()).resolves.toEqual({ value: undefined, done: true });
    await expect.poll(() => disconnected, { timeout: 1000 }).toBe(true);
  });

  it("yields all events on ordinary completion", async () => {
    const client = new RestfulClient({ baseUrl, paths: { events: "/complete" } });
    const received: string[] = [];
    for await (const event of client.stream("events", { method: "GET" })) {
      received.push(event.data);
    }
    expect(received).toEqual(["first"]);
  });
});
