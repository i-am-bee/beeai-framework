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

  it("Serializes", async () => {
    const tool = new OpenAPITool({ openApiSchema });

    const serialized = await tool.serialize();
    const deserialized = await OpenAPITool.fromSerialized(serialized);
    verifyDeserialization(tool, deserialized);
  });
});
