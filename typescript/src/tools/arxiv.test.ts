/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ArXivTool } from "@/tools/arxiv.js";
import { ToolInput } from "@/tools/base.js";

// A real response of `https://export.arxiv.org/api/query?id_list=1706.03762,2301.10000&max_results=2`
const response = readFileSync(
  fileURLToPath(new URL("./__fixtures__/arxiv-response.xml", import.meta.url)),
  "utf8",
);

describe("ArXivTool", () => {
  let instance: ArXivTool;

  beforeEach(() => {
    instance = new ArXivTool();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Response parsing", () => {
    const parse = () => instance["_parseResponse"](new Response(response, { status: 200 }));

    it("Parses the feed metadata", async () => {
      expect(await parse()).toMatchObject({ totalResults: 2, startIndex: 0, itemsPerPage: 2 });
    });

    it("Extracts the id without the URL prefix and the version", async () => {
      const { entries } = await parse();
      expect(entries.map((entry) => entry.id)).toStrictEqual(["1706.03762", "2301.10000"]);
    });

    it("Keeps the URL of the entry", async () => {
      const { entries } = await parse();
      expect(entries.map((entry) => (entry as { url?: string }).url)).toStrictEqual([
        "http://arxiv.org/abs/1706.03762v7",
        "http://arxiv.org/abs/2301.10000v6",
      ]);
    });

    it("Extracts the categories", async () => {
      const { entries } = await parse();
      expect(entries.map((entry) => entry.primaryCategory)).toStrictEqual(["cs.CL", "astro-ph.CO"]);
      expect(entries.map((entry) => entry.categories)).toStrictEqual([
        ["cs.CL", "cs.LG"],
        ["astro-ph.CO", "gr-qc", "hep-ph", "hep-th"],
      ]);
    });

    it("Extracts the links", async () => {
      const { entries } = await parse();
      expect(entries.map((entry) => entry.links)).toStrictEqual([
        ["https://arxiv.org/abs/1706.03762v7", "https://arxiv.org/pdf/1706.03762v7"],
        [
          "https://arxiv.org/abs/2301.10000v6",
          "https://arxiv.org/pdf/2301.10000v6",
          "https://doi.org/10.1140/epjc/s10052-024-13218-2",
        ],
      ]);
    });

    it("Extracts the remaining fields", async () => {
      const { entries } = await parse();
      expect(entries[0]).toMatchObject({
        title: "Attention Is All You Need",
        published: "2017-06-12T17:57:34Z",
        updated: "2023-08-02T00:41:18Z",
        comment: "15 pages, 5 figures",
      });
      expect(entries[0].authors).toHaveLength(8);
      expect(entries[0].authors[0]).toStrictEqual({ name: "Ashish Vaswani", affiliation: [] });
      expect(entries[1]).toMatchObject({
        doi: "10.1140/epjc/s10052-024-13218-2",
        journalReference: "Eur. Phys. J. C 84 (2024) 884",
      });
      expect(entries[1].authors.map((author) => author.name)).toStrictEqual([
        "Sayantan Choudhury",
        "Mayukh R. Gangopadhyay",
        "M. Sami",
      ]);
    });
  });

  describe("Query building", () => {
    const prepare = (input: Partial<ToolInput<ArXivTool>>) =>
      instance["_prepareQuery"]({ start: 0, maxResults: 5, ...input });

    it("Builds a query from the filters", () => {
      expect(
        prepare({
          search_query: {
            include: [
              { field: "title", value: "attention is all" },
              { field: "author", value: "Vaswani" },
            ],
            exclude: [{ field: "abstract", value: "vision" }],
          },
        }),
      ).toBe(
        "start=0&max_results=5&search_query=ti:attention+is+all+AND+au:Vaswani+ANDNOT+abs:vision&sortBy=relevance&sortOrder=descending",
      );
    });

    it("Encodes the reserved characters of the values", () => {
      const query = prepare({
        search_query: { include: [{ field: "title", value: "AT&T R#D 100%" }] },
      });
      expect(query).toBe(
        "start=0&max_results=5&search_query=ti:AT%26T+R%23D+100%25&sortBy=relevance&sortOrder=descending",
      );

      const params = new URL(`https://export.arxiv.org/api/query?${query}`).searchParams;
      expect(params.get("search_query")).toBe("ti:AT&T R#D 100%");
      expect(params.get("sortBy")).toBe("relevance");
    });

    it("Does not turn a plus sign of a value into a space", () => {
      const query = prepare({
        search_query: { include: [{ field: "title", value: "C++" }] },
      });
      const params = new URL(`https://export.arxiv.org/api/query?${query}`).searchParams;
      expect(params.get("search_query")).toBe("ti:C++");
    });

    it.each([
      ["https://arxiv.org/abs/1706.03762v7", "1706.03762"],
      ["http://arxiv.org/abs/1706.03762v7", "1706.03762"],
      ["http://arxiv.org/abs/1706.03762v12", "1706.03762"],
      ["https://arxiv.org/pdf/2301.10000v123", "2301.10000"],
      ["http://arxiv.org/abs/hep-th/9901001v10", "hep-th/9901001"],
      ["2301.10000v6", "2301.10000"],
      ["2301.10000", "2301.10000"],
    ])("Normalizes the id %s", (id, expected) => {
      const params = new URL(`https://export.arxiv.org/api/query?${prepare({ ids: [id] })}`)
        .searchParams;
      expect(params.get("id_list")).toBe(expected);
    });

    it("Sends multiple ids as a single comma-separated list", () => {
      const query = prepare({
        ids: ["http://arxiv.org/abs/1706.03762v7", "https://arxiv.org/pdf/2301.10000"],
      });
      expect(query).toBe(
        "start=0&max_results=5&id_list=1706.03762,2301.10000&sortBy=relevance&sortOrder=descending",
      );
    });
  });

  it("Calls the API with the encoded query and parses the response", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(response, { status: 200 }));

    const output = await instance.run({
      search_query: { include: [{ field: "title", value: "AT&T R#D 100%" }] },
      maxResults: 2,
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "https://export.arxiv.org/api/query?start=0&max_results=2&search_query=ti:AT%26T+R%23D+100%25&sortBy=relevance&sortOrder=descending",
    );
    expect(output.isEmpty()).toBe(false);
    expect(output.result.entries.map((entry) => entry.id)).toStrictEqual([
      "1706.03762",
      "2301.10000",
    ]);
  });
});
