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

import { PathLike } from "fs";
import * as crypto from "node:crypto";
import { pipeline } from "node:stream/promises";
import fs, { createReadStream } from "node:fs";
import { copyFile } from "node:fs/promises";
import path from "node:path";
import { Cache } from "@/cache/decoratorCache.js";
import { Serializable } from "@/internals/serializable.js";
import { shallowCopy } from "@/serializer/utils.js";

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

    await Promise.all(
      files.map((file) =>
        copyFile(
          path.join(this.input.interpreterWorkingDir.toString(), file.pythonId),
          path.join(this.input.localWorkingDir.toString(), file.filename),
        ),
      ),
    );
    return files.map((file) => ({ ...file, id: file.pythonId }));
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
