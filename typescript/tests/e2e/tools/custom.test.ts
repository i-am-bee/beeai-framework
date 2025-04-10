/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import { describe, it, expect } from "vitest";
import { CustomTool, CustomToolCreateError } from "@/tools/custom.js";
import { StringToolOutput } from "@/tools/base.js";

describe.runIf(process.env.CODE_INTERPRETER_URL)("CustomTool", () => {
  it("should instantiate correctly", async () => {
    const customTool = await CustomTool.fromSourceCode(
      { url: process.env.CODE_INTERPRETER_URL! },
      `
      def test_func(a: int, b: str=None):
        """A test tool"""
        print(a)
        print(b)
      `,
    );

    expect(customTool.name).toBe("test_func");
    expect(customTool.description).toBe("A test tool");
    expect(await customTool.inputSchema()).toEqual({
      $schema: "http://json-schema.org/draft-07/schema#",
      additionalProperties: false,
      properties: {
        a: { type: "integer" },
        b: { type: "string" },
      },
      required: ["a"],
      title: "test_func",
      type: "object",
    });
  });

  it("should throw InvalidCustomToolError on parse error", async () => {
    await expect(
      CustomTool.fromSourceCode({ url: process.env.CODE_INTERPRETER_URL! }, "source code"),
    ).rejects.toThrowError(CustomToolCreateError);
  });

  it("should run the custom tool", async () => {
    const customTool = await CustomTool.fromSourceCode(
      { url: process.env.CODE_INTERPRETER_URL! },
      `import requests
  
def get_riddle(a: int, b: str) -> dict[str, str] | None:
  """
  Fetches a random riddle from the Riddles API.

  This function retrieves a random riddle and its answer. Testing with input params.

  Returns:
      dict[str,str] | None: A dictionary containing:
          - 'riddle' (str): Passed in a int
          - 'answer' (str): Passed in b string
      Returns None if the request fails.
  """
  return { "riddle": str(a), "answer": b}`,
    );

    const result = await customTool.run(
      {
        a: 42,
        b: "something",
      },
      {
        signal: new AbortController().signal,
      },
    );
    expect(result).toBeInstanceOf(StringToolOutput);
    expect(result.getTextContent()).toEqual('{"riddle": "42", "answer": "something"}');
  });

  it("should throw CustomToolExecutionError on execution error", async () => {
    const customTool = await CustomTool.fromSourceCode(
      { url: process.env.CODE_INTERPRETER_URL! },
      `
def test(a: int, b: str=None):
    """A test tool"""
    print("hello")
    div_by_zero = 123 / 0
    return "foo"
`,
    );

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
});
