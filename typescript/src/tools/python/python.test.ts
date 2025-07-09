/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from "vitest";
import { PythonTool } from "@/tools/python/python.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";
import { LocalPythonStorage } from "@/tools/python/storage.js";

const codeInterpreterUrl = process.env.CODE_INTERPRETER_URL || "http://localhost:50081";

const getPythonTool = () =>
  new PythonTool({
    codeInterpreter: { url: codeInterpreterUrl },
    storage: new LocalPythonStorage({
      interpreterWorkingDir: "/tmp/code-interpreter-storage",
      localWorkingDir: "./test_dir/",
    }),
  });

describe("PythonTool", () => {
  it("Is the expected tool", () => {
    const tool = getPythonTool();
    expect(tool).toBeInstanceOf(PythonTool);
    expect(PythonTool.isTool(tool)).toBe(true);
    expect(tool.name).toBe("Python");
    expect(tool.description).toMatch("Run Python and/or shell code");
  });

  it("Throws input validation error for wrong language", async () => {
    await expect(
      getPythonTool().run({
        // @ts-ignore
        language: "PL/1",

        code: "# won't get this far because we don't support PL/1 yet",
        inputFiles: [],
      }),
    ).rejects.toThrow("The received tool input does not match the expected schema.");
  });

  it("Throws input validation error for missing file", async () => {
    const sourceCode = `
    with open("test_file.txt", 'r') as f:
        print(f.read())
    `;

    await expect(
      getPythonTool().run({
        language: "python",
        code: sourceCode,
        inputFiles: ["bogus_file.txt"],
      }),
    ).rejects.toThrow("The received tool input does not match the expected schema.");
  });

  it("serializes", async () => {
    const tool = getPythonTool();
    const serialized = await tool.serialize();
    const deserializedTool = await PythonTool.fromSerialized(serialized);
    verifyDeserialization(tool, deserializedTool);
  });
});
