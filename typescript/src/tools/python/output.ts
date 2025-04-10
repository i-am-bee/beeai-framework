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

import { ToolOutput } from "@/tools/base.js";
import { PythonFile } from "@/tools/python/storage.js";

export class PythonToolOutput extends ToolOutput {
  static FILE_PREFIX = "urn:bee:file";

  constructor(
    public readonly stdout: string,
    public readonly stderr: string,
    public readonly exitCode?: number,
    public readonly outputFiles: PythonFile[] = [],
  ) {
    super();
  }

  static {
    this.register();
  }

  isEmpty() {
    return false;
  }

  getTextContent() {
    const executionStatus =
      this.exitCode === 0
        ? "The code executed successfully."
        : `The code exited with error code ${this.exitCode}.`;
    const stdout = this.stdout.trim() ? `Standard output: \n\`\`\`\n${this.stdout}\n\`\`\`` : null;
    const stderr = this.stderr.trim() ? `Error output: \n\`\`\`\n${this.stderr}\n\`\`\`` : null;
    const isImage = (filename: string) =>
      [".png", ".jpg", ".jpeg", ".gif", ".bmp"].some((ext) => filename.toLowerCase().endsWith(ext));
    const files = this.outputFiles.length
      ? "The following files were created or modified. The user does not see them yet. To present a file to the user, send them the link below, verbatim:\n" +
        this.outputFiles
          .map(
            (file) =>
              `${isImage(file.filename) ? "!" : ""}[${file.filename}](${PythonToolOutput.FILE_PREFIX}:${file.id})`,
          )
          .join("\n")
      : null;

    return [executionStatus, stdout, stderr, files].filter(Boolean).join("\n");
  }

  createSnapshot() {
    return {
      stdout: this.stdout,
      stderr: this.stderr,
      exitCode: this.exitCode,
      outputFiles: this.outputFiles,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
