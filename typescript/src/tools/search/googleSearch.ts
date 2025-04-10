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

import { customsearch_v1 as GoogleSearchAPI } from "@googleapis/customsearch";
import {
  SearchToolOptions,
  SearchToolOutput,
  SearchToolResult,
  SearchToolRunOptions,
} from "./base.js";
import { ToolEmitter, Tool, ToolInput } from "@/tools/base.js";
import { z } from "zod";
import { Cache } from "@/cache/decoratorCache.js";
import { ValueError } from "@/errors.js";
import { parseEnv } from "@/internals/env.js";
import { RunContext } from "@/context.js";
import { paginate } from "@/internals/helpers/paginate.js";
import { ValidationError } from "ajv";
import { Emitter } from "@/emitter/emitter.js";

export interface GoogleSearchToolOptions extends SearchToolOptions {
  apiKey?: string;
  cseId?: string;
  maxResults: number;
}

type GoogleSearchToolRunOptions = SearchToolRunOptions;

export interface GoogleSearchToolResult extends SearchToolResult {}

export class GoogleSearchToolOutput extends SearchToolOutput<GoogleSearchToolResult> {
  constructor(public readonly results: GoogleSearchToolResult[]) {
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

export class GoogleSearchTool extends Tool<
  GoogleSearchToolOutput,
  GoogleSearchToolOptions,
  GoogleSearchToolRunOptions
> {
  name = "GoogleSearch";
  description = `Search for online trends, news, current events, real-time information, or research topics.`;

  public readonly emitter: ToolEmitter<ToolInput<this>, GoogleSearchToolOutput> =
    Emitter.root.child({
      namespace: ["tool", "search", "google"],
      creator: this,
    });

  @Cache()
  inputSchema() {
    return z.object({
      query: z.string({ description: `Search query` }).min(1).max(2048),
    });
  }

  protected apiKey: string;
  protected cseId: string;

  public constructor(options: GoogleSearchToolOptions = { maxResults: 10 }) {
    super(options);

    this.apiKey = options.apiKey || parseEnv("GOOGLE_API_KEY", z.string());
    this.cseId = options.cseId || parseEnv("GOOGLE_CSE_ID", z.string());

    if (!this.apiKey || !this.cseId) {
      throw new ValueError(
        [
          `"apiKey" or "cseId" must both be provided.`,
          `Either set them directly or put them in ENV ("GOOGLE_API_KEY" / "GOOGLE_CSE_ID")`,
        ].join("\n"),
      );
    }

    if (options.maxResults < 1 || options.maxResults > 100) {
      throw new ValidationError([
        {
          message: "Property 'maxResults' must be between 1 and 100",
          propertyName: "options.maxResults",
        },
      ]);
    }
  }

  @Cache()
  protected get client(): GoogleSearchAPI.Customsearch {
    return new GoogleSearchAPI.Customsearch({
      auth: this.apiKey,
    });
  }

  static {
    this.register();
  }

  protected async _run(
    { query: input }: ToolInput<this>,
    _options: Partial<GoogleSearchToolRunOptions>,
    run: RunContext<this>,
  ) {
    const results = await paginate({
      size: this.options.maxResults,
      handler: async ({ cursor = 0, limit }) => {
        const maxChunkSize = 10;

        const {
          data: { items = [] },
        } = await this.client.cse.list(
          {
            cx: this.cseId,
            q: input,
            start: cursor,
            num: Math.min(limit, maxChunkSize),
            safe: "active",
          },
          {
            signal: run.signal,
          },
        );

        return {
          data: items,
          nextCursor: items.length < maxChunkSize ? undefined : cursor + items.length,
        };
      },
    });

    return new GoogleSearchToolOutput(
      results.map((result) => ({
        title: result.title || "",
        description: result.snippet || "",
        url: result.link || "",
      })),
    );
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      apiKey: this.apiKey,
      cseId: this.cseId,
    };
  }

  loadSnapshot({ apiKey, cseId, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    super.loadSnapshot(snapshot);
    Object.assign(this, {
      apiKey,
      cseId,
    });
  }
}
