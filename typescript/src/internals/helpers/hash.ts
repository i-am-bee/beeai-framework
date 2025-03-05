/**
 * Copyright 2025 IBM Corp.
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
