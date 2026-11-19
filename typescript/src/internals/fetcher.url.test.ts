/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { RestfulClient } from "@/internals/fetcher.js";

describe("RestfulClient URL joining", () => {
  let server: Server;
  let origin: string;

  beforeEach(async () => {
    server = createServer((request, response) => {
      if (!["/api/echo", "/api", "/api/"].includes(request.url!)) {
        response.statusCode = 404;
      }
      response.end(JSON.stringify({ path: request.url }));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it.each([
    { base: "/api", path: "echo", expected: "/api/echo" },
    { base: "/api", path: "/echo", expected: "/api/echo" },
    { base: "/api/", path: "echo", expected: "/api/echo" },
    { base: "/api/", path: "/echo", expected: "/api/echo" },
    { base: "/api", path: "", expected: "/api" },
    { base: "/api/", path: "", expected: "/api/" },
  ])("joins $base and $path", async ({ base, path, expected }) => {
    const client = new RestfulClient({ baseUrl: origin + base, paths: { echo: path } });
    expect(await client.fetch("echo")).toEqual({ path: expected });
  });
});
