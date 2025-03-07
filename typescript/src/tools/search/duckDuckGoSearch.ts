/**
 * Copyright 2025 IBM Corp.
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

import { SearchOptions, search as rawDDGSearch, SafeSearchType } from "duck-duck-scrape";
import { Options as ThrottleOptions } from "p-throttle";
import {
  SearchToolOptions,
  SearchToolOutput,
  SearchToolResult,
  SearchToolRunOptions,
} from "./base.js";
import { ToolEmitter, Tool, ToolInput } from "@/tools/base.js";
import { HeaderGenerator } from "header-generator";
import type { NeedleOptions } from "needle";
import { z } from "zod";
import { Cache } from "@/cache/decoratorCache.js";
import { RunContext } from "@/context.js";
import { paginate } from "@/internals/helpers/paginate.js";
import { Emitter } from "@/emitter/emitter.js";

export { SafeSearchType as DuckDuckGoSearchToolSearchType };

export interface DuckDuckGoSearchToolOptions extends SearchToolOptions {
  search?: SearchOptions;
  throttle?: ThrottleOptions | false;
  httpClientOptions?: NeedleOptions;
  maxResults: number;
}

export interface DuckDuckGoSearchToolRunOptions extends SearchToolRunOptions {
  search?: SearchOptions;
  httpClientOptions?: NeedleOptions;
}

export interface DuckDuckGoSearchToolResult extends SearchToolResult {}

export class DuckDuckGoSearchToolOutput extends SearchToolOutput<DuckDuckGoSearchToolResult> {
  constructor(public readonly results: DuckDuckGoSearchToolResult[]) {
    super(results);
  }

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

export class DuckDuckGoSearchTool extends Tool<
  DuckDuckGoSearchToolOutput,
  DuckDuckGoSearchToolOptions,
  DuckDuckGoSearchToolRunOptions
> {
  name = "DuckDuckGo";
  description =
    "Search for online trends, news, current events, real-time information, or research topics.";

  public readonly emitter: ToolEmitter<ToolInput<this>, DuckDuckGoSearchToolOutput> =
    Emitter.root.child({
      namespace: ["tool", "search", "ddg"],
      creator: this,
    });

  @Cache()
  inputSchema() {
    return z.object({
      query: z.string({ description: `Search query` }).min(1).max(128),
    });
  }

  public constructor(options: Partial<DuckDuckGoSearchToolOptions> = {}) {
    super({ ...options, maxResults: options?.maxResults ?? 15 });
  }

  static {
    this.register();
  }

  @Cache({ enumerable: false })
  protected async _createClient() {
    const { throttle } = this.options;
    if (throttle === false) {
      return rawDDGSearch;
    }

    const { default: pThrottle } = await import("p-throttle");
    return pThrottle({
      ...throttle,
      limit: throttle?.limit ?? 1,
      interval: throttle?.interval ?? 3000,
    })(rawDDGSearch);
  }

  protected async _run(
    { query: input }: ToolInput<this>,
    options: Partial<DuckDuckGoSearchToolRunOptions>,
    run: RunContext<this>,
  ) {
    const headers = new HeaderGenerator().getHeaders();
    const client = await this._createClient();

    const results = await paginate({
      size: this.options.maxResults,
      handler: async ({ cursor = 0 }) => {
        const { results: data, noResults: done } = await client(
          input,
          {
            safeSearch: SafeSearchType.MODERATE,
            ...this.options.search,
            ...options?.search,
            offset: cursor,
          },
          {
            headers,
            user_agent: headers["user-agent"],
            ...this.options?.httpClientOptions,
            ...options?.httpClientOptions,
            signal: run.signal,
            uri_modifier: (rawUrl) => {
              const url = new URL(rawUrl);
              url.searchParams.delete("ss_mkt");
              return url.toString();
            },
          },
        );

        return {
          data,
          nextCursor: done ? undefined : cursor + data.length,
        };
      },
    });

    const { stripHtml } = await import("string-strip-html");

    return new DuckDuckGoSearchToolOutput(
      results.map((result) => ({
        title: stripHtml(result.title).result,
        description: stripHtml(result.description).result,
        url: result.url,
      })),
    );
  }
}
