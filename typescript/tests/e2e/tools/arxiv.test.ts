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
