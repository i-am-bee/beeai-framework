/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  DuckDuckGoSearchTool,
  DuckDuckGoSearchToolOptions,
  DuckDuckGoSearchToolOutput,
} from "@/tools/search/duckDuckGoSearch.js";
import { beforeEach, expect, vi } from "vitest";

import * as ddg from "duck-duck-scrape";
import { Tool } from "@/tools/base.js";
import { Task } from "promise-based-task";

import { SlidingCache } from "@/cache/slidingCache.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";
vi.mock("duck-duck-scrape");

describe("DuckDuckGoSearch Tool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.clearAllTimers();
  });

  const generateResults = (count = 3): ddg.SearchResults => {
    return {
      noResults: count === 0,
      vqd: "",
      results: Array(count)
        .fill(null)
        .map((_, i) => ({
          title: `Result ${i + 1}`,
          url: `https://example.com/#${i + 1}`,
          description: `<h1>Response ${i + 1}</h1>`,
          rawDescription: `<h1>Response ${i + 1}</h1>`,
          hostname: "hostname",
          icon: "",
          bang: undefined,
        })),
    };
  };

  it("Is valid tool", async () => {
    const tool = new DuckDuckGoSearchTool();
    expect(tool).instanceOf(Tool);
    expect(Tool.isTool(tool)).toBe(true);
    expect(tool.name).toBeDefined();
    expect(tool.description).toBeDefined();
  });

  interface RetrieveDataInput {
    query: string;
    options: DuckDuckGoSearchToolOptions;
  }
  it.each([
    { query: "LLM", options: { maxResults: 1 } },
    { query: "IBM Research" },
    { query: "NLP", options: { maxResults: 3 } },
  ] as RetrieveDataInput[])("Retrieves data (%o)", async (input) => {
    const globalMaxResults = 10;
    const maxResultsPerPage = (input as any).options?.maxResultsPerPage ?? globalMaxResults;

    const tool = new DuckDuckGoSearchTool({
      maxResults: globalMaxResults,
      cache: false,
      throttle: false,
    });

    vi.mocked(ddg.search).mockResolvedValue(generateResults(maxResultsPerPage));

    const response = await tool.run({ query: input.query }, input.options);
    expect(response.results.length).toEqual(maxResultsPerPage);
    expect(ddg.search).toBeCalledWith(
      input.query,
      expect.objectContaining(input.options?.search ?? {}),
      expect.any(Object),
    );
  });

  it("Serializes", async () => {
    const tool = new DuckDuckGoSearchTool({
      throttle: false,
      cache: new SlidingCache({
        size: 10,
        ttl: 1000,
      }),
      maxResults: 1,
    });

    await tool.cache!.set(
      "A",
      Task.resolve(
        new DuckDuckGoSearchToolOutput([
          {
            title: "A",
            url: "http://example.com",
            description: "A",
          },
        ]),
      ),
    );
    await tool.cache!.set("B", Task.resolve(new DuckDuckGoSearchToolOutput([])));
    const serialized = await tool.serialize();
    const deserialized = await DuckDuckGoSearchTool.fromSerialized(serialized);
    expect(await tool.cache.get("A")).toStrictEqual(await deserialized.cache.get("A"));
    verifyDeserialization(tool, deserialized);
  });
});
