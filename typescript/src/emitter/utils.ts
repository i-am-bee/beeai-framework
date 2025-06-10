/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
