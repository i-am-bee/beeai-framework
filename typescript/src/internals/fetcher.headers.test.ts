/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { RestfulClient } from "@/internals/fetcher.js";

describe("RestfulClient headers", () => {
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    server = createServer((request, response) => {
      response.end(JSON.stringify({ header: request.headers["x-custom"] ?? null }));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  const variants = [
    { name: "tuples", make: (): HeadersInit => [["X-Custom", "present"]], expected: "present" },
    {
      name: "duplicate tuples",
      make: (): HeadersInit => [
        ["X-Custom", "first"],
        ["X-Custom", "second"],
      ],
      expected: "first, second",
    },
    { name: "record", make: (): HeadersInit => ({ "X-Custom": "present" }), expected: "present" },
    {
      name: "Headers",
      make: (): HeadersInit => new Headers({ "X-Custom": "present" }),
      expected: "present",
    },
  ];

  describe.each(["default", "request"])("%s headers", (location) => {
    it.each(variants)("sends $name headers to the server", async ({ make, expected }) => {
      const headers = make();
      const client = new RestfulClient({
        baseUrl,
        paths: { echo: "/echo" },
        headers: location === "default" ? async () => headers : undefined,
      });

      const result = await client.fetch("echo", location === "request" ? { headers } : {});
      expect(result).toEqual({ header: expected });
    });
  });
});
