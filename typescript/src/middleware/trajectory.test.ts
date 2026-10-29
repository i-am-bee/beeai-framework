/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { GlobalTrajectoryMiddleware } from "@/middleware/trajectory.js";

class Level {
  public readonly emitter = new Emitter();
  constructor(public readonly name: string) {}
}

describe("GlobalTrajectoryMiddleware", () => {
  it("Logs runs nested several levels deep once per event", async () => {
    const outer = new Level("outer");
    const middle = new Level("middle");
    const inner = new Level("inner");
    const lines: string[] = [];

    try {
      await RunContext.enter(outer, { params: [] }, async () =>
        RunContext.enter(middle, { params: [] }, async () =>
          RunContext.enter(inner, { params: [] }, async () => "done"),
        ),
      ).middleware(new GlobalTrajectoryMiddleware({ target: (line) => lines.push(line) }));

      expect(lines.some((line) => line.includes("Level[inner][start]"))).toBe(true);
      expect(lines.length).toBe(new Set(lines).size);
    } finally {
      [outer, middle, inner].forEach((level) => level.emitter.destroy());
    }
  });
});
