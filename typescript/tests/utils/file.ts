/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { createRandomHash } from "@/internals/helpers/hash.js";

export async function getTempFile(name: string, data: string) {
  const fullPath = path.join(os.tmpdir(), `${createRandomHash(4)}_${name}`);
  await fs.promises.writeFile(fullPath, data);

  return {
    fullPath,
    async [Symbol.asyncDispose]() {
      await fs.promises.unlink(fullPath);
    },
  };
}
