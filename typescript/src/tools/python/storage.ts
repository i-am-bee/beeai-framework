/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { PathLike } from "fs";
import * as crypto from "node:crypto";
import { pipeline } from "node:stream/promises";
import fs, { createReadStream } from "node:fs";
import { copyFile } from "node:fs/promises";
import path from "node:path";
import { Cache } from "@/cache/decoratorCache.js";
import { Serializable } from "@/internals/serializable.js";
import { shallowCopy } from "@/serializer/utils.js";
import { ToolError } from "@/tools/base.js";

export interface PythonFile {
  id: string;
  pythonId: string;
  filename: string;
}

export interface PythonUploadFile {
  id: string;
  filename: string;
}

export interface PythonDownloadFile {
  id?: string;
  filename: string;
  pythonId: string;
}

export abstract class PythonStorage extends Serializable {
  /**
   * List all files that code interpreter can use.
   */
  abstract list(): Promise<PythonFile[]>;

  /**
   * Prepare subset of available files to code interpreter.
   */
  abstract upload(files: PythonUploadFile[]): Promise<PythonFile[]>;

  /**
   * Process updated/modified/deleted files from code interpreter response.
   */
  abstract download(files: PythonDownloadFile[]): Promise<PythonFile[]>;
}

export class TemporaryStorage extends PythonStorage {
  protected files: PythonFile[] = [];

  async list() {
    return this.files.slice();
  }

  async upload(files: PythonUploadFile[]): Promise<PythonFile[]> {
    return files.map((file) => ({
      id: file.id,
      pythonId: file.id,
      filename: file.filename,
    }));
  }

  async download(files: PythonDownloadFile[]) {
    this.files = [
      ...this.files.filter((file) => files.every((f) => f.filename !== file.filename)),
      ...files.map((file) => ({ id: file.pythonId, ...file })),
    ];
    return this.files.slice();
  }

  createSnapshot() {
    return { files: this.files.slice() };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

interface Input {
  localWorkingDir: PathLike;
  interpreterWorkingDir: PathLike;
  ignoredFiles?: Set<string>;
}

export class LocalPythonStorage extends PythonStorage {
  protected readonly input: Required<Input>;

  constructor(input: Input) {
    super();
    this.input = {
      ...input,
      ignoredFiles: input?.ignoredFiles ?? new Set([".gitkeep"]),
    };
  }

  @Cache()
  protected async init() {
    await fs.promises.mkdir(this.input.localWorkingDir, { recursive: true });
    await fs.promises.mkdir(this.input.interpreterWorkingDir, { recursive: true });
  }

  async list(): Promise<PythonFile[]> {
    await this.init();

    const files = await fs.promises.readdir(this.input.localWorkingDir, {
      withFileTypes: true,
      recursive: false,
    });
    return Promise.all(
      files
        .filter((file) => file.isFile() && !this.input.ignoredFiles.has(file.name))
        .map(async (file) => {
          const pythonId = await this.computeHash(
            path.join(this.input.localWorkingDir.toString(), file.name),
          );

          return {
            id: pythonId,
            filename: file.name,
            pythonId,
          };
        }),
    );
  }

  async upload(files: PythonUploadFile[]): Promise<PythonFile[]> {
    await this.init();

    const fileList = await this.list();

    await Promise.all(
      files.map((file) => {
        const filesystemFile = fileList.find((filesystemFile) => filesystemFile.id === file.id)!;
        return copyFile(
          path.join(this.input.localWorkingDir.toString(), filesystemFile.filename),
          path.join(this.input.interpreterWorkingDir.toString(), filesystemFile.id),
        );
      }),
    );
    return files.map((file) => ({ ...file, pythonId: file.id }));
  }

  async download(files: PythonDownloadFile[]) {
    await this.init();

    // Both names come from the code interpreter response; check every path before any copy starts.
    const copies = await Promise.all(
      files.map(async (file) => ({
        source: await LocalPythonStorage.resolveInside(
          this.input.interpreterWorkingDir,
          file.pythonId,
        ),
        target: await LocalPythonStorage.resolveInside(this.input.localWorkingDir, file.filename),
      })),
    );
    await Promise.all(copies.map(({ source, target }) => copyFile(source, target)));
    return files.map((file) => ({ ...file, id: file.pythonId }));
  }

  /**
   * Resolve `name` against `baseDir`, following symlinks, and ensure the result is strictly inside
   * `baseDir`. When the path does not exist yet, its parent directory is resolved instead.
   */
  protected static async resolveInside(baseDir: PathLike, name: string): Promise<string> {
    const base = await fs.promises.realpath(baseDir.toString());
    const lexical = path.resolve(base, name);
    LocalPythonStorage.assertInside(base, lexical, name);
    let resolved: string;
    try {
      resolved = await fs.promises.realpath(lexical);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
        throw err;
      }
      // copyFile would follow a dangling symlink to wherever it points, so refuse it.
      const isLink = await fs.promises.lstat(lexical).then(
        (stats) => stats.isSymbolicLink(),
        () => false,
      );
      if (isLink) {
        throw new ToolError(`Path traversal detected: ${name}`);
      }
      const parent = await fs.promises.realpath(path.dirname(lexical));
      resolved = path.join(parent, path.basename(lexical));
    }
    LocalPythonStorage.assertInside(base, resolved, name);
    return resolved;
  }

  protected static assertInside(base: string, target: string, name: string): void {
    const relative = path.relative(base, target);
    if (
      !relative ||
      relative === ".." ||
      relative.startsWith(`..${path.sep}`) ||
      path.isAbsolute(relative)
    ) {
      throw new ToolError(`Path traversal detected: ${name}`);
    }
  }

  protected async computeHash(file: PathLike) {
    const hash = crypto.createHash("sha256");
    await pipeline(createReadStream(file), hash);
    return hash.digest("hex");
  }

  createSnapshot() {
    return { input: shallowCopy(this.input) };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
