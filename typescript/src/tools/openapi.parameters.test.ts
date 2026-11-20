/**
 * Copyright 2026 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OpenAPITool } from "@/tools/openapi.js";

const tools: OpenAPITool[] = [];
const fetch = vi.fn<typeof globalThis.fetch>();

function schema(parameters: any[] = []) {
  return {
    openapi: "3.0.3",
    info: { title: "PetsAPI", version: "1" },
    servers: [{ url: "https://example.invalid/api?configured=keep" }],
    paths: { "/pets": { get: { parameters, responses: { "200": { description: "Pets" } } } } },
  };
}

function createTool(spec: object = schema(), fetchOptions?: RequestInit) {
  const tool = new OpenAPITool({ openApiSchema: JSON.stringify(spec), fetchOptions });
  tools.push(tool);
  return tool;
}

function request(index = 0) {
  const [url, options] = fetch.mock.calls[index];
  return { url: new URL(String(url)), headers: new Headers(options?.headers) };
}

beforeEach(() => {
  fetch.mockReset().mockImplementation(() => Promise.resolve(new Response("ok")));
  vi.stubGlobal("fetch", fetch);
});

afterEach(() => {
  vi.unstubAllGlobals();
  tools.splice(0).forEach((tool) => tool.emitter.destroy());
});

describe("OpenAPITool parameter locations", () => {
  it("inherits Path Item parameters and lets the operation override them", async () => {
    const tool = createTool({
      ...schema(),
      paths: {
        "/pets/{petId}": {
          parameters: [
            { name: "petId", in: "path", required: true, schema: { type: "string" } },
            { name: "X-Region", in: "header", schema: { type: "string" } },
            { name: "theme", in: "cookie", schema: { type: "string" } },
            { name: "verbose", in: "query", schema: { type: "string" } },
          ],
          get: {
            parameters: [{ name: "verbose", in: "query", schema: { type: "boolean" } }],
            responses: { "200": { description: "Pet" } },
          },
        },
      },
    });

    await tool.run({
      path: "/pets/{petId}",
      method: "get",
      parameters: { "petId": "a/b", "X-Region": "east", "theme": "dark", "verbose": false },
    });

    const { url, headers } = request();
    expect(url.toString()).toBe(
      "https://example.invalid/api/pets/a%2Fb?configured=keep&verbose=false",
    );
    expect(headers.get("X-Region")).toBe("east");
    expect(headers.get("Cookie")).toBe("theme=dark");
  });

  it("inherits referenced parameters and resolves required operation overrides", async () => {
    const tool = createTool({
      ...schema(),
      components: {
        parameters: {
          Region: { name: "X-Region", in: "header", schema: { type: "string" } },
          RequiredRegion: {
            name: "X-Region",
            in: "header",
            required: true,
            schema: { type: "string", enum: ["east"] },
          },
        },
      },
      paths: {
        "/pets": {
          parameters: [{ $ref: "#/components/parameters/Region" }],
          get: {
            parameters: [{ $ref: "#/components/parameters/RequiredRegion" }],
            responses: { "200": { description: "Pets" } },
          },
        },
      },
    });
    await expect(tool.run({ path: "/pets", method: "get", parameters: {} })).rejects.toThrow();
    expect(fetch).not.toHaveBeenCalled();
    await tool.run({ path: "/pets", method: "get", parameters: { "X-Region": "east" } });
    expect(request().headers.get("X-Region")).toBe("east");
  });

  it.each([
    ["summary", "Pets"],
    ["description", "Pet operations"],
    ["servers", [{ url: "https://example.invalid" }]],
    ["x-owner", { team: "pets" }],
  ])("does not expose Path Item %s as an operation", (key, value) => {
    const spec = schema();
    const tool = createTool({
      ...spec,
      paths: { "/pets": { ...spec.paths["/pets"], [key as string]: value } },
    });
    expect(tool.inputSchema().oneOf).toMatchObject([{ properties: { method: { const: "get" } } }]);
  });

  it.each(["header", "cookie"])(
    "sends a %s parameter in its declared location",
    async (location) => {
      const name = location === "header" ? "X-Region" : "region";
      const tool = createTool(schema([{ name, in: location, schema: { type: "string" } }]));
      await tool.run({ path: "/pets", method: "get", parameters: { [name]: "east" } });

      const { url, headers } = request();
      expect(headers.get(location === "header" ? name : "Cookie")).toBe(
        location === "header" ? "east" : "region=east",
      );
      expect(url.search).toBe("?configured=keep");
    },
  );

  it("resolves the locations of referenced header and cookie parameters", async () => {
    const spec = {
      ...schema([
        { $ref: "#/components/parameters/Region" },
        { $ref: "#/components/parameters/Theme" },
      ]),
      components: {
        parameters: {
          Region: { name: "X-Region", in: "header", schema: { type: "string" } },
          Theme: { name: "theme", in: "cookie", schema: { type: "string" } },
        },
      },
    };
    await createTool(spec).run({
      path: "/pets",
      method: "get",
      parameters: { "X-Region": "east", "theme": "dark" },
    });

    const { url, headers } = request();
    expect(headers.get("X-Region")).toBe("east");
    expect(headers.get("Cookie")).toBe("theme=dark");
    expect(url.search).toBe("?configured=keep");
  });

  it("keeps encoded path parameters, query parameters and the configured base URL", async () => {
    const spec = schema();
    const tool = createTool({
      ...spec,
      paths: {
        "/pets/{petId}": {
          get: {
            parameters: [
              { name: "petId", in: "path", required: true, schema: { type: "string" } },
              { name: "verbose", in: "query", schema: { type: "boolean" } },
            ],
            responses: { "200": { description: "Pet" } },
          },
        },
      },
    });
    await tool.run({
      path: "/pets/{petId}",
      method: "get",
      parameters: { petId: "a/b c", verbose: false },
    });
    expect(request().url.toString()).toBe(
      "https://example.invalid/api/pets/a%2Fb%20c?configured=keep&verbose=false",
    );
  });

  it("uses the parameter locations of the selected operation", async () => {
    const spec = schema([{ name: "region", in: "header", schema: { type: "string" } }]);
    const tool = createTool({
      ...spec,
      paths: {
        "/pets": {
          ...spec.paths["/pets"],
          post: {
            parameters: [{ name: "region", in: "cookie", schema: { type: "string" } }],
            responses: { "200": { description: "Pets" } },
          },
        },
      },
    });
    await tool.run({ path: "/pets", method: "get", parameters: { region: "east" } });
    await tool.run({ path: "/pets", method: "post", parameters: { region: "west" } });
    expect(request(0).headers.get("region")).toBe("east");
    expect(request(0).headers.get("Cookie")).toBeNull();
    expect(request(1).headers.get("region")).toBeNull();
    expect(request(1).headers.get("Cookie")).toBe("region=west");
  });

  it("preserves configured headers case-insensitively", async () => {
    const tool = createTool(
      schema([
        { name: "X-Region", in: "header", schema: { type: "string" } },
        { name: "X-Trace", in: "header", schema: { type: "string" } },
      ]),
      { headers: new Headers({ "x-region": "configured" }) },
    );
    await tool.run({
      path: "/pets",
      method: "get",
      parameters: { "X-Region": "east", "X-Trace": "trace" },
    });
    expect(request().headers.get("X-Region")).toBe("configured");
    expect(request().headers.get("X-Trace")).toBe("trace");
    expect(request().url.search).toBe("?configured=keep");
  });

  it("ignores reserved OpenAPI header parameters", async () => {
    const tool = createTool(
      schema(
        ["Accept", "Content-Type", "Authorization"].map((name) => ({
          name,
          in: "header",
          schema: { type: "string" },
        })),
      ),
      { headers: { Authorization: "configured" } },
    );
    await tool.run({
      path: "/pets",
      method: "get",
      parameters: {
        "Accept": "unused",
        "Content-Type": "unused",
        "Authorization": "unused",
      },
    });
    expect(request().headers.get("Accept")).toBe("application/json");
    expect(request().headers.get("Content-Type")).toBeNull();
    expect(request().headers.get("Authorization")).toBe("configured");
    expect(request().url.search).toBe("?configured=keep");
  });

  it("encodes cookie values and keeps configured cookies", async () => {
    const tool = createTool(
      schema([
        { name: "region", in: "cookie", schema: { type: "string" } },
        { name: "theme", in: "cookie", schema: { type: "string" } },
      ]),
      { headers: { Cookie: "configured=keep" } },
    );
    await tool.run({
      path: "/pets",
      method: "get",
      parameters: { region: "east; vip=true", theme: "dark" },
    });
    expect(request().headers.get("Cookie")).toBe(
      "configured=keep; region=east%3B%20vip%3Dtrue; theme=dark",
    );
    expect(request().url.search).toBe("?configured=keep");
  });

  it("serializes array header parameters with the default simple style", async () => {
    const tool = createTool(
      schema([
        { name: "X-Tags", in: "header", schema: { type: "array", items: { type: "string" } } },
      ]),
    );
    await tool.run({ path: "/pets", method: "get", parameters: { "X-Tags": ["a", "b"] } });
    expect(request().headers.get("X-Tags")).toBe("a,b");
  });

  it.each([".", ".."])('continues to reject the path parameter "%s"', async (petId) => {
    const spec = schema();
    const tool = createTool({
      ...spec,
      paths: {
        "/pets/{petId}": {
          get: {
            parameters: [{ name: "petId", in: "path", required: true, schema: { type: "string" } }],
            responses: { "200": { description: "Pet" } },
          },
        },
      },
    });
    await expect(
      tool.run({ path: "/pets/{petId}", method: "get", parameters: { petId } }),
    ).rejects.toThrow("Invalid value for path parameter");
    expect(fetch).not.toHaveBeenCalled();
  });
});
