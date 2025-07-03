/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseToolOptions, BaseToolRunOptions, ToolOutput } from "@/tools/base.js";
import { Cache, WeakRefKeyFn } from "@/cache/decoratorCache.js";
import * as R from "remeda";

export interface SearchToolOptions extends BaseToolOptions {}

export interface SearchToolRunOptions extends BaseToolRunOptions {}

export interface SearchToolResult {
  title: string;
  description: string;
  url: string;
}

export abstract class SearchToolOutput<
  TSearchToolResult extends SearchToolResult = SearchToolResult,
> extends ToolOutput {
  constructor(public readonly results: TSearchToolResult[]) {
    super();
  }

  @Cache({
    cacheKey: WeakRefKeyFn.from<SearchToolOutput>((self) => self.results),
    enumerable: false,
  })
  get sources() {
    return R.unique(this.results.map((result) => result.url));
  }

  isEmpty() {
    return this.results.length === 0;
  }

  @Cache({
    cacheKey: WeakRefKeyFn.from<SearchToolOutput>((self) => self.results),
  })
  getTextContent(): string {
    return this.results.map((result) => JSON.stringify(result, null, 2)).join("\n\n");
  }
}
