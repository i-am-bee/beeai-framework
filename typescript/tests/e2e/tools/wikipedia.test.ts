/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { WikipediaTool } from "@/tools/search/wikipedia.js";
import { expect } from "vitest";

describe("Wikipedia", () => {
  it("Retrieves data", async () => {
    const instance = new WikipediaTool();
    const response = await instance.run({ query: "Molecule" });

    expect(response.results).toHaveLength(1);
    const result = response.results[0];
    expect(result).toBeTruthy();
    expect(result).toMatchObject({
      title: expect.any(String),
      description: expect.any(String),
      url: expect.any(String),
      fields: expect.any(Object),
    });

    const markdown = response.results[0].fields!.markdown;
    expect(markdown).toBeTruthy();
    expect(markdown).not.match(/\[([^\]]+)\]\(([^)]+)\)/g);
  });

  it("Handles non-existing page", async () => {
    const instance = new WikipediaTool();
    const response = await instance.run({ query: "adsdassdsadas" });

    expect(response.isEmpty()).toBeTruthy();
    expect(response.results).toHaveLength(0);
    expect(response.getTextContent()).toMatchInlineSnapshot(
      `"No results were found. Try to reformat your query."`,
    );
  });
});
