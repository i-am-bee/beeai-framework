/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { takeBigger } from "@/internals/helpers/number.js";

describe("takeBigger", () => {
  it("returns the biggest number", () => {
    expect(takeBigger(3, 7, 5)).toBe(7);
    expect(takeBigger(7)).toBe(7);
    expect(takeBigger(-3, -1)).toBe(-1);
  });

  it("ignores undefined and null", () => {
    expect(takeBigger(undefined, 4, null)).toBe(4);
    expect(takeBigger(2, undefined)).toBe(2);
  });

  it("returns 0 without any numeric operand", () => {
    expect(takeBigger()).toBe(0);
    expect(takeBigger(undefined)).toBe(0);
    expect(takeBigger(undefined, null)).toBe(0);
    expect(takeBigger(null, undefined, null)).toBe(0);
  });

  it("returns 0 for NaN", () => {
    expect(takeBigger(Number.NaN)).toBe(0);
  });
});
