/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { pRetry } from "@/internals/helpers/retry.js";

const delays = vi.hoisted(() => [] as number[]);

vi.mock("node:timers/promises", () => ({
  setTimeout: vi.fn(async (ms: number) => {
    delays.push(ms);
  }),
}));

describe("pRetry", () => {
  beforeEach(() => {
    delays.length = 0;
  });

  const failingAttempts = async (factor: number | undefined, retries: number) => {
    const fn = vi.fn(async () => {
      throw new Error("Failed");
    });
    await expect(pRetry(fn, { factor, retries })).rejects.toThrowError("Failed");
    expect(fn).toHaveBeenCalledTimes(retries + 1);
    return delays;
  };

  it.each([
    { factor: undefined, expected: [2000, 4000, 8000, 16000] },
    { factor: 2, expected: [2000, 4000, 8000, 16000] },
    { factor: 3, expected: [3000, 9000, 27000, 81000] },
    { factor: 1, expected: [1000, 1000, 1000, 1000] },
    { factor: 1.5, expected: [1500, 2250, 3375, 5063] },
    { factor: 0.5, expected: [500, 250, 125, 63] },
    { factor: 0, expected: [0, 0, 0, 0] },
  ])("Waits factor^(attempt - 1) seconds between attempts (factor: $factor)", async (input) => {
    expect(await failingAttempts(input.factor, 4)).toStrictEqual(input.expected);
  });

  it("Does not wait before the first attempt", async () => {
    expect(await pRetry(async () => "ok", { retries: 3 })).toBe("ok");
    expect(delays).toStrictEqual([]);
  });
});
