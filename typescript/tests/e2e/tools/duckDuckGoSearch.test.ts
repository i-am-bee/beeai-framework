/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect } from "vitest";
import { DuckDuckGoSearchTool } from "@/tools/search/duckDuckGoSearch.js";

describe.skip("DuckDuckGo", () => {
  it("Retrieves data", async () => {
    const instance = new DuckDuckGoSearchTool();
    const response = await instance.run({ query: "BeeAI Framework" });
    expect(response.results.length).toBeGreaterThan(0);
  });
});
