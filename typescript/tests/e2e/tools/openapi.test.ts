/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, expect } from "vitest";
import { OpenAPITool } from "@/tools/openapi.js";

describe("OpenAPITool", () => {
  let instance: OpenAPITool;

  // Simple API spec for a cat fact API
  // Assisted by WCA@IBM
  // Latest GenAI contribution: ibm/granite-20b-code-instruct-v2

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

  beforeEach(() => {
    instance = new OpenAPITool({
      openApiSchema,
    });
  });

  it("Runs", async () => {
    const response = await instance.run(
      {
        path: "/fact",
        method: "get",
      },
      {
        signal: AbortSignal.timeout(60 * 1000),
        retryOptions: {},
      },
    );

    expect(response.isEmpty()).toBe(false);
  });
});
