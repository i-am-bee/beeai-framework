/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
