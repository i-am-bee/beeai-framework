/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { FileCache } from "@/cache/fileCache.js";
import { afterEach, beforeEach, vi } from "vitest";

describe("FileCache", () => {
  let directory: string;

  beforeEach(async () => {
    directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), "beeai-file-cache-"));
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await fs.promises.rm(directory, { recursive: true, force: true });
  });

  it("waits for the file write before resolving set", async () => {
    const fullPath = path.join(directory, "cache.json");
    const writeFile = fs.promises.writeFile.bind(fs.promises);
    let releaseWrite!: () => void;
    const writeGate = new Promise<void>((resolve) => {
      releaseWrite = resolve;
    });
    const writeSpy = vi.spyOn(fs.promises, "writeFile").mockImplementation(async (...args) => {
      await writeGate;
      return writeFile(...args);
    });

    const cache = new FileCache<number>({ fullPath });
    let settled = false;
    const setPromise = cache.set("key", 42).then(() => {
      settled = true;
    });

    await vi.waitFor(() => expect(writeSpy).toHaveBeenCalledOnce());
    expect(settled).toBe(false);

    releaseWrite();
    await setPromise;
    await expect(new FileCache<number>({ fullPath }).get("key")).resolves.toBe(42);
  });

  it("reports a failed file write to the caller", async () => {
    const cache = new FileCache<number>({
      fullPath: path.join(directory, "missing", "cache.json"),
    });

    await expect(cache.set("key", 42)).rejects.toMatchObject({ code: "ENOENT" });
  });
});
