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
