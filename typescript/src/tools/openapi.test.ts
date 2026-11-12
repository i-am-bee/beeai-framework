/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAPITool } from "@/tools/openapi.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";

const openApiSchema =
  '{\
    "openapi": "3.0.0",\
    "info": {\
      "title": "Cat Facts API",\
      "description": "A simple API for cat facts",\
      "version": "1.0.0"\
    },\
    "servers": [\
      {\
        "url": "https://catfact.ninja",\
        "description": "Production server"\
      }\
    ],\
    "paths": {\
      "/fact": {\
        "get": {\
          "summary": "Get a random cat fact",\
          "description": "Returns a random cat fact.",\
          "responses": {\
            "200": {\
              "description": "Successful response",\
              "content": {\
                "application/json": {\
                  "schema": {\
                    "$ref": "#/components/schemas/Fact"\
                  }\
                }\
              }\
            }\
          }\
        }\
      }\
    },\
    "components": {\
      "schemas": {\
        "Fact": {\
          "type": "object",\
          "properties": {\
            "fact": {\
              "type": "string",\
              "description": "The cat fact"\
            }\
          }\
        }\
      }\
    }\
  }';

describe("OpenAPI Tool", () => {
  beforeEach(() => {
    vi.clearAllTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("Serializes", async () => {
    const tool = new OpenAPITool({ openApiSchema });

    const serialized = await tool.serialize();
    const deserialized = await OpenAPITool.fromSerialized(serialized, {
      allowFunctionDeserialization: true,
    });
    verifyDeserialization(tool, deserialized);
  });

  it("sends JSON request bodies with the correct content type", async () => {
    const fetchMock = vi.fn(async (_url: string, _options: RequestInit) => new Response("ok"));
    vi.stubGlobal("fetch", fetchMock);
    const schema = JSON.stringify({
      openapi: "3.0.0",
      info: { title: "Test API", version: "1.0.0" },
      servers: [{ url: "https://example.com" }],
      paths: {
        "/items": {
          post: {
            requestBody: {
              content: {
                "application/json": {
                  schema: { type: "object", properties: { name: { type: "string" } } },
                },
              },
            },
          },
        },
      },
    });
    const input = { path: "/items", method: "post", body: { name: "Bee" } };

    await new OpenAPITool({ openApiSchema: schema }).run(input);
    expect(fetchMock).toHaveBeenCalledOnce();
    const options = fetchMock.mock.calls[0][1] as RequestInit;
    expect(options.body).toBe('{"name":"Bee"}');
    expect(new Headers(options.headers).get("Content-Type")).toBe("application/json");

    await new OpenAPITool({
      openApiSchema: schema,
      fetchOptions: {
        headers: new Headers({ "Content-Type": "application/vnd.api+json", "X-Test": "1" }),
      },
    }).run(input);
    const customOptions = fetchMock.mock.calls[1][1] as RequestInit;
    expect(customOptions.body).toBe('{"name":"Bee"}');
    expect(new Headers(customOptions.headers).get("Content-Type")).toBe("application/vnd.api+json");
    expect(new Headers(customOptions.headers).get("X-Test")).toBe("1");
  });

  describe("path parameters", () => {
    const operation = (type: string) => ({
      get: { parameters: [{ name: "id", in: "path", required: true, schema: { type } }] },
    });
    const pathParamSchema = JSON.stringify({
      openapi: "3.0.0",
      info: { title: "Test API", version: "1.0.0" },
      servers: [{ url: "https://example.com/api" }],
      paths: {
        "/users/{id}": operation("string"),
        "/items/{id}": operation("integer"),
        "/pairs/{id}/{id}": operation("string"),
        "/dots/{id}..": operation("string"),
      },
    });

    const callTool = async (path: string, id: unknown) => {
      const fetchMock = vi.fn(async (_url: string, _options: RequestInit) => new Response("ok"));
      vi.stubGlobal("fetch", fetchMock);
      const result = new OpenAPITool({ openApiSchema: pathParamSchema }).run({
        path,
        method: "get",
        parameters: { id },
      });
      return { fetchMock, result };
    };

    const fetchedUrl = async (path: string, id: unknown) => {
      const { fetchMock, result } = await callTool(path, id);
      await result;
      return new URL(fetchMock.mock.calls[0][0]);
    };

    it("substitutes a plain value under the server base path", async () => {
      expect((await fetchedUrl("/users/{id}", "42")).pathname).toBe("/api/users/42");
    });

    it("substitutes a number", async () => {
      expect((await fetchedUrl("/items/{id}", 42)).pathname).toBe("/api/items/42");
    });

    it("fills every occurrence of a placeholder", async () => {
      expect((await fetchedUrl("/pairs/{id}/{id}", "a")).pathname).toBe("/api/pairs/a/a");
    });

    it("percent-encodes values so they stay one path segment", async () => {
      const url = await fetchedUrl("/users/{id}", "../../admin?x=1#y");
      expect(url.pathname).toBe("/api/users/..%2F..%2Fadmin%3Fx%3D1%23y");
      expect(url.search).toBe("");
    });

    it.each([".", ".."])("rejects the dot segment %j", async (id) => {
      const { fetchMock, result } = await callTool("/users/{id}", id);
      await expect(result).rejects.toThrow(`Invalid value for path parameter 'id'`);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("rejects a value that completes a dot segment with the template", async () => {
      const { fetchMock, result } = await callTool("/dots/{id}..", "");
      await expect(result).rejects.toThrow("Invalid path after substituting parameters");
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });
});
