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

import { expect } from "vitest";
import { PythonTool } from "@/tools/python/python.js";
import { LocalPythonStorage } from "@/tools/python/storage.js";

import { ToolError } from "@/tools/base.js";

const getPythonTool = () =>
  new PythonTool({
    codeInterpreter: { url: process.env.CODE_INTERPRETER_URL! },
    storage: new LocalPythonStorage({
      interpreterWorkingDir: "/tmp/code-interpreter-storage",
      localWorkingDir: "./test_dir/",
    }),
  });

describe.runIf(process.env.CODE_INTERPRETER_URL)("PythonTool", () => {
  it("Returns zero exitCode and stdout results", async () => {
    const result = await getPythonTool().run({
      language: "python",
      code: "print('hello')",
    });

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("hello\n");
  });

  it("Returns non-zero exitCode and stderr for bad python", async () => {
    const result = await getPythonTool().run({
      language: "python",
      code: "PUT LIST (((ARR(I,J) DO I = 1 TO 5) DO J = 1 TO 5))",
    });

    expect(result.exitCode).toBe(1);
    expect(result.stderr).toMatch("SyntaxError");
  });

  it("Throws tool error for code exceptions", async () => {
    const sourceCode = `
    with open("wrong_file_here.txt", 'r') as f:
        print(f.read())
    `;

    await expect(
      getPythonTool().run({
        language: "python",
        code: sourceCode,
        inputFiles: ["test_file.txt"],
      }),
    ).rejects.toThrowError(ToolError);
  });
});
