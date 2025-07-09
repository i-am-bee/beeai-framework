/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

declare const __LIBRARY_VERSION: string;

let Version = "0.0.0";
try {
  Version = __LIBRARY_VERSION;
} catch {
  /* empty */
}

export { Version };
