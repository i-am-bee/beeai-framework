/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { CalculatorTool } from "@/tools/calculator.js";
import { beforeEach, expect } from "vitest";

describe("Calculator", () => {
  let instance: CalculatorTool;

  beforeEach(() => {
    instance = new CalculatorTool();
  });

  it("Runs", async () => {
    const x1 = 1;
    const y1 = 1;
    const x2 = 4;
    const y2 = 5;

    const response = await instance.run({
      expression: `sqrt( (${x2}-${x1})^2 + (${y2}-${y1})^2 )`,
    });
    expect(response.result).toBe(5);
  });

  it("Throws", async () => {
    await expect(
      instance.run({
        expression: "import",
      }),
    ).rejects.toThrowError();
  });
});
