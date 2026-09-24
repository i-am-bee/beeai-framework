/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { cosineSimilarityMatrix } from "@/internals/helpers/math.js";

describe("cosineSimilarityMatrix", () => {
  it("rejects matrices with differing column counts", () => {
    expect(() => cosineSimilarityMatrix([[1, 2]], [[1, 2, 3]])).toThrowError(
      "Matrices must have the same number of columns.",
    );
    expect(() => cosineSimilarityMatrix([[1, 2]], [])).toThrowError(
      "Matrices must have the same number of columns.",
    );
  });

  it("computes similarity for matching column counts", () => {
    const result = cosineSimilarityMatrix([[1, 0]], [[1, 0]]);
    expect(result[0][0]).toBeCloseTo(1);
  });
});
