/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { LocalPythonStorage } from "@/tools/python/storage.js";

describe("LocalPythonStorage.download", () => {
  let root: string;
  let localWorkingDir: string;
  let interpreterWorkingDir: string;
  let storage: LocalPythonStorage;

  beforeEach(async () => {
    // Working directories sit two levels below the temporary root, so escaped writes stay inside it.
    root = await fs.promises.mkdtemp(path.join(os.tmpdir(), "beeai-storage-"));
    localWorkingDir = path.join(root, "work", "local");
    interpreterWorkingDir = path.join(root, "work", "interpreter");
    storage = new LocalPythonStorage({ localWorkingDir, interpreterWorkingDir });
    await fs.promises.mkdir(interpreterWorkingDir, { recursive: true });
    await fs.promises.writeFile(path.join(interpreterWorkingDir, "abc123"), "result");
    await fs.promises.writeFile(path.join(root, "work", "outside"), "secret");
  });

  afterEach(async () => {
    await fs.promises.rm(root, { recursive: true, force: true });
  });

  it("copies an interpreter file into the local working directory", async () => {
    await storage.download([{ filename: "result.txt", pythonId: "abc123" }]);

    await expect(
      fs.promises.readFile(path.join(localWorkingDir, "result.txt"), "utf8"),
    ).resolves.toBe("result");
  });

  it("allows a file name that only starts with two dots", async () => {
    await storage.download([{ filename: "..result.txt", pythonId: "abc123" }]);

    await expect(
      fs.promises.readFile(path.join(localWorkingDir, "..result.txt"), "utf8"),
    ).resolves.toBe("result");
  });

  it.each([
    ["one level up", "../escaped.txt"],
    ["two levels up", "../../escaped.txt"],
    ["absolute", "<root>/escaped.txt"],
    ["empty", ""],
    ["current directory", "."],
  ])("rejects a file name outside the local working directory (%s)", async (_case, name) => {
    const filename = name.replace("<root>", root);
    await expect(storage.download([{ filename, pythonId: "abc123" }])).rejects.toThrow(
      "Path traversal detected",
    );
    expect(fs.existsSync(path.join(root, "escaped.txt"))).toBe(false);
    expect(fs.existsSync(path.join(root, "work", "escaped.txt"))).toBe(false);
  });

  it("rejects an interpreter file id outside the interpreter working directory", async () => {
    await expect(
      storage.download([{ filename: "copied.txt", pythonId: "../outside" }]),
    ).rejects.toThrow("Path traversal detected");
    expect(fs.existsSync(path.join(localWorkingDir, "copied.txt"))).toBe(false);
  });

  it("checks every file before copying any", async () => {
    await expect(
      storage.download([
        { filename: "valid.txt", pythonId: "abc123" },
        { filename: "../escaped.txt", pythonId: "abc123" },
      ]),
    ).rejects.toThrow("Path traversal detected");
    expect(fs.existsSync(path.join(localWorkingDir, "valid.txt"))).toBe(false);
  });

  describe("symlinks", () => {
    const outsideDir = () => path.join(root, "work", "elsewhere");

    beforeEach(async () => {
      await fs.promises.mkdir(localWorkingDir, { recursive: true });
      await fs.promises.mkdir(outsideDir(), { recursive: true });
    });

    it("rejects an interpreter file id that links outside", async () => {
      await fs.promises.symlink(
        path.join(root, "work", "outside"),
        path.join(interpreterWorkingDir, "link"),
      );

      await expect(
        storage.download([{ filename: "copied.txt", pythonId: "link" }]),
      ).rejects.toThrow("Path traversal detected");
      expect(fs.existsSync(path.join(localWorkingDir, "copied.txt"))).toBe(false);
    });

    it("rejects a file name under a directory that links outside", async () => {
      await fs.promises.symlink(outsideDir(), path.join(localWorkingDir, "linked"));

      await expect(
        storage.download([{ filename: "linked/escaped.txt", pythonId: "abc123" }]),
      ).rejects.toThrow("Path traversal detected");
      expect(fs.existsSync(path.join(outsideDir(), "escaped.txt"))).toBe(false);
    });

    it("rejects a file name that is a dangling link to outside", async () => {
      await fs.promises.symlink(
        path.join(outsideDir(), "escaped.txt"),
        path.join(localWorkingDir, "dangling"),
      );

      await expect(
        storage.download([{ filename: "dangling", pythonId: "abc123" }]),
      ).rejects.toThrow("Path traversal detected");
      expect(fs.existsSync(path.join(outsideDir(), "escaped.txt"))).toBe(false);
    });

    it("accepts a nested file name inside the working directory", async () => {
      await fs.promises.mkdir(path.join(localWorkingDir, "out"));

      await storage.download([{ filename: "out/result.txt", pythonId: "abc123" }]);

      await expect(
        fs.promises.readFile(path.join(localWorkingDir, "out", "result.txt"), "utf8"),
      ).resolves.toBe("result");
    });
  });
});
