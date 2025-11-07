/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash as _createHash, randomBytes } from "node:crypto";
import { NotImplementedError } from "@/errors.js";

export function createHash(input: string, length = 4) {
  if (length > 32) {
    throw new NotImplementedError("Max supported hash length is 32");
  }

  return _createHash("sha256")
    .update(input)
    .digest("hex")
    .slice(0, length * 2);
}

export function createRandomHash(length = 4) {
  return createHash(randomBytes(20).toString("base64"), length);
}
