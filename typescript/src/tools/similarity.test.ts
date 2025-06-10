/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, expect, vi } from "vitest";

import { Tool } from "@/tools/base.js";
import { SimilarityTool, SimilarityToolOptions } from "./similarity.js";
vi.mock("duck-duck-scrape");

describe("WebSearch Tool", () => {
  beforeEach(() => {
    vi.clearAllTimers();
  });

  const similarityProvider: SimilarityToolOptions<unknown>["provider"] = async ({ documents }) =>
    documents.map((_, idx) => ({ score: idx }));

  it("Is valid tool", async () => {
    const tool = new SimilarityTool({ provider: similarityProvider });
    expect(tool).instanceOf(Tool);
    expect(Tool.isTool(tool)).toBe(true);
    expect(tool.name).toBeDefined();
    expect(tool.description).toBeDefined();
  });

  it("Uses provider", async () => {
    const providerMock = vi.fn(similarityProvider);
    const tool = new SimilarityTool({ provider: providerMock });

    await tool.run({
      query: "foo",
      documents: [{ text: "foo" }, { text: "bar" }],
    });

    expect(providerMock).toBeCalled();
  });

  it("Returns top-k results", async () => {
    const providerMock = vi.fn(similarityProvider);

    const maxResultsGlobal = 3;
    const maxResultsRun = 2;

    const tool = new SimilarityTool({ provider: providerMock, maxResults: maxResultsGlobal });

    const output1 = await tool.run({
      query: "foo",
      documents: [{ text: "foo" }, { text: "bar" }, { text: "foo" }, { text: "bar" }],
    });

    expect(output1.result.length).toBe(maxResultsGlobal);

    const output2 = await tool.run(
      {
        query: "foo",
        documents: [{ text: "foo" }, { text: "bar" }, { text: "foo" }, { text: "bar" }],
      },
      { maxResults: maxResultsRun },
    );
    expect(output2.result.length).toBe(maxResultsRun);
  });
});
