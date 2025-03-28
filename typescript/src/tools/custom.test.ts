/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, it, expect, vi } from "vitest";
import { CustomTool } from "./custom.js";
import { StringToolOutput } from "./base.js";

const mocks = vi.hoisted(() => {
  return {
    fetch: vi.fn(),
  };
});

vi.stubGlobal("fetch", mocks.fetch);

describe("CustomTool", () => {
  it("should instantiate correctly", async () => {
    mocks.fetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tool_name: "test",
          tool_description: "A test tool",
          tool_input_schema_json: JSON.stringify({
            $schema: "http://json-schema.org/draft-07/schema#",
            type: "object",
            properties: {
              a: { type: "integer" },
              b: { type: "string" },
            },
          }),
        }),
    });

    const customTool = await CustomTool.fromSourceCode({ url: "http://localhost" }, "source code");

    expect(customTool.name).toBe("test");
    expect(customTool.description).toBe("A test tool");
    expect(await customTool.inputSchema()).toEqual({
      $schema: "http://json-schema.org/draft-07/schema#",
      type: "object",
      properties: {
        a: { type: "integer" },
        b: { type: "string" },
      },
    });
  });

  it("should throw CustomToolCreateError on parse error", async () => {
    mocks.fetch.mockRejectedValueOnce({
      cause: { name: "HTTPParserError" },
    });

    await expect(
      CustomTool.fromSourceCode({ url: "http://localhost" }, "source code"),
    ).rejects.toThrow(
      "Request to code interpreter has failed -- ensure that CODE_INTERPRETER_URL points to the new HTTP endpoint",
    );
  });

  it("should run the custom tool", async () => {
    mocks.fetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tool_name: "test",
          tool_description: "A test tool",
          tool_input_schema_json: JSON.stringify({
            $schema: "http://json-schema.org/draft-07/schema#",
            type: "object",
            properties: {
              a: { type: "integer" },
              b: { type: "string" },
            },
          }),
        }),
    });

    const customTool = await CustomTool.fromSourceCode({ url: "http://localhost" }, "source code");

    mocks.fetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          exit_code: 0,
          tool_output_json: '{"something": "42"}',
        }),
    });

    const result = await customTool.run(
      {
        a: 42,
        b: "test",
      },
      {
        signal: new AbortController().signal,
      },
    );
    expect(result).toBeInstanceOf(StringToolOutput);
    expect(result.getTextContent()).toEqual('{"something": "42"}');
  });

  it("should throw CustomToolExecuteError on execution error", async () => {
    mocks.fetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tool_name: "test",
          tool_description: "A test tool",
          tool_input_schema_json: JSON.stringify({
            $schema: "http://json-schema.org/draft-07/schema#",
            type: "object",
            properties: {
              a: { type: "integer" },
              b: { type: "string" },
            },
          }),
        }),
    });

    const customTool = await CustomTool.fromSourceCode({ url: "http://localhost" }, "source code");

    mocks.fetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          stderr: "Oh no, it does not work",
        }),
    });

    await expect(
      customTool.run(
        {
          a: 42,
          b: "test",
        },
        {
          signal: new AbortController().signal,
        },
      ),
    ).rejects.toThrow('Tool "test" has occurred an error!');
  });

  it("should handle HTTP error responses", async () => {
    mocks.fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Internal Server Error"),
    });

    await expect(
      CustomTool.fromSourceCode({ url: "http://localhost" }, "source code"),
    ).rejects.toThrow("Request to code interpreter has failed with HTTP status code 500");
  });
});
