/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { RestfulClient } from "@/internals/fetcher.js";

describe("RestfulClient query parameters", () => {
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    server = createServer((request, response) => {
      const query = new URL(request.url!, "http://localhost").searchParams;
      response.end(JSON.stringify({ tags: query.getAll("tag"), keep: query.get("keep") }));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api?tag=old&tag=older&keep=base`;
  });

  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it.each([
    { values: ["first", "second"], expected: ["first", "second"] },
    { values: ["", "second"], expected: ["", "second"] },
    { values: ["only"], expected: ["only"] },
    { values: [], expected: ["old", "older"] },
  ])("preserves values $values on the wire", async ({ values, expected }) => {
    const client = new RestfulClient({ baseUrl, paths: { echo: "/echo" } });
    const searchParams = new URLSearchParams(values.map((value) => ["tag", value]));

    expect(await client.fetch("echo", { searchParams })).toEqual({ tags: expected, keep: "base" });
    expect(searchParams.getAll("tag")).toEqual(values);
  });
});
