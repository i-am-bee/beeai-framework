/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ArXivResponse, ArXivTool } from "@/tools/arxiv.js";
import { beforeEach, expect } from "vitest";
import { ToolError } from "@/tools/base.js";

describe("Arxiv", () => {
  let instance: ArXivTool;

  beforeEach(() => {
    instance = new ArXivTool();
  });

  it("Runs", async () => {
    const response = await instance.run(
      {
        search_query: {
          include: [
            {
              value: "LLM",
              field: "title",
            },
          ],
        },
        maxResults: 2,
      },
      {
        signal: AbortSignal.timeout(60 * 1000),
        retryOptions: {},
      },
    );

    expect(response.isEmpty()).toBe(false);
    expect(response.result.startIndex).toBe(0);
    expect(response.result.entries.length).toBe(2);
    for (const entry of response.result.entries) {
      expect(entry).toMatchObject({
        id: expect.any(String),
        title: expect.any(String),
        summary: expect.any(String),
        published: expect.any(String),
      } as ArXivResponse["entries"][0]);
    }
  });

  it("Throws", async () => {
    await expect(
      instance.run({
        ids: ["xx"],
      }),
    ).rejects.toThrowError(new ToolError(`Request to ArXiv API has failed!`));
  });
});
