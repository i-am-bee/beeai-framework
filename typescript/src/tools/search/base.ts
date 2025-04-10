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
