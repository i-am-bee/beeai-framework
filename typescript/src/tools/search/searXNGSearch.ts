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

import { ToolEmitter, Tool, ToolInput } from "@/tools/base.js";
import { z } from "zod";
import { Emitter } from "@/emitter/emitter.js";
import { createURLParams } from "@/internals/fetcher.js";
import { RunContext } from "@/context.js";
import {
  SearchToolOptions,
  SearchToolOutput,
  SearchToolResult,
  SearchToolRunOptions,
} from "@/tools/search/base.js";
import { ValidationError } from "ajv";
import { parseEnv } from "@/internals/env.js";

export interface SearXNGToolOptions extends SearchToolOptions {
  baseUrl?: string;
  maxResults: number;
}

type SearXNGToolRunOptions = SearchToolRunOptions;

export interface SearXNGToolResult extends SearchToolResult {}

export class SearXNGToolOutput extends SearchToolOutput<SearXNGToolResult> {
  static {
    this.register();
  }

  createSnapshot() {
    return {
      results: this.results,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

export class SearXNGTool extends Tool<
  SearXNGToolOutput,
  SearXNGToolOptions,
  SearXNGToolRunOptions
> {
  name = "Web Search";
  description = `Search for online trends, news, current events, real-time information, or research topics.`;

  public readonly emitter: ToolEmitter<ToolInput<this>, SearchToolOutput<SearXNGToolResult>> =
    Emitter.root.child({
      namespace: ["tool", "search", "searXNG"],
      creator: this,
    });

  inputSchema() {
    return z.object({
      query: z.string().min(1).describe(`Web search query.`),
    });
  }

  public constructor(options: SearXNGToolOptions = { maxResults: 10 }) {
    super(options);

    if (options.maxResults < 1 || options.maxResults > 100) {
      throw new ValidationError([
        {
          message: "Property 'maxResults' must be between 1 and 100",
          propertyName: "options.maxResults",
        },
      ]);
    }
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<SearchToolRunOptions>,
    run: RunContext<this>,
  ) {
    const params = createURLParams({
      q: input.query,
      format: "json",
    });

    const baseUrl = this.options.baseUrl || parseEnv("SEARXNG_BASE_URL", z.string());
    const url = `${baseUrl}?${decodeURIComponent(params.toString())}`;
    const response = await fetch(url, {
      signal: run.signal,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const data = await response.json();

    return new SearXNGToolOutput(
      data.results
        .slice(0, this.options.maxResults)
        .map((result: { url: string; title: string; content: string }) => ({
          url: result.url || "",
          title: result.title || "",
          description: result.content || "",
        })),
    );
  }
}
