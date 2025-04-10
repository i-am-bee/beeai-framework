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

import { EmitterError } from "@/emitter/errors.js";

export function assertValidNamespace(path: string[]) {
  if (!path || !Array.isArray(path)) {
    throw new EmitterError("Event path cannot be empty!");
  }
  path.forEach((part) => assertValidName(part));
}

export function assertValidName(name: string) {
  if (!name || !/(^[a-zA-Z0-9_]+$)/.test(name)) {
    throw new EmitterError(
      "Event name or a namespace part must contain only letters, letters and optionally underscores.",
      [],
      {
        context: { value: name },
      },
    );
  }
}

export function createFullPath(path: string[], name: string) {
  return name ? path.concat(name).join(".") : path.join(".");
}

export function isPath(name: string) {
  return name.includes(".");
}
