/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { setTimeout } from "node:timers/promises";
import { ZodType } from "zod";

export async function* sleep(ms: number, signal?: AbortSignal) {
  const start = Date.now();
  const ctx = { iteration: 0, elapsed: 0 };
  while (true) {
    await setTimeout(ms, {
      signal,
    });
    ctx.elapsed = Date.now() - start;
    ctx.iteration++;
    yield ctx;
  }
}

export function validate<T>(value: unknown, schema: ZodType<T>): asserts value is T {
  schema.parse(value);
}
