/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
