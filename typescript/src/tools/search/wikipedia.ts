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

import type { Page, pageFunctions, searchOptions } from "wikipedia";
import wiki from "wikipedia";
import { Cache } from "@/cache/decoratorCache.js";
import * as R from "remeda";
import { keys, mapValues } from "remeda";
import { ArrayKeys, Common } from "@/internals/types.js";
import {
  SearchToolOptions,
  SearchToolOutput,
  SearchToolResult,
  SearchToolRunOptions,
} from "./base.js";
import { asyncProperties } from "@/internals/helpers/promise.js";
import { z } from "zod";
import { Tool, ToolEmitter, ToolInput } from "@/tools/base.js";
import Turndown from "turndown";
// @ts-expect-error missing types
import turndownPlugin from "joplin-turndown-plugin-gfm";
import stringComparison from "string-comparison";
import { pageResult } from "wikipedia/dist/resultTypes.js";
import { Emitter } from "@/emitter/emitter.js";

wiki.default.setLang("en");

export interface SearchOptions extends searchOptions {}

export interface FilterOptions {
  excludeOthersOnExactMatch?: boolean;
  minPageNameSimilarity?: number;
}

export type PageFunctions = Record<
  pageFunctions,
  {
    transform?: <T>(output: T) => T;
  }
> & {
  markdown: {
    transform?: (output: string) => string;
    filter?: (node: HTMLElement) => boolean;
  };
};

export interface ExtractionOptions {
  fields?: Partial<PageFunctions>;
}

export interface OutputOptions {
  maxSerializedLength?: number;
  maxDescriptionLength?: number;
}

export interface WikipediaToolOptions extends SearchToolOptions {
  filters?: FilterOptions;
  search?: SearchOptions;
  extraction?: ExtractionOptions;
  output?: OutputOptions;
}

export interface WikipediaToolRunOptions extends SearchToolRunOptions {
  filters?: FilterOptions;
  search?: SearchOptions;
  extraction?: ExtractionOptions;
  output?: OutputOptions;
}

type PageWithMarkdown = Page & { markdown: () => Promise<string> };

type ResultFields = { [K in keyof PageFunctions]: Awaited<ReturnType<PageWithMarkdown[K]>> };

export interface WikipediaToolResult extends SearchToolResult {
  fields: Partial<ResultFields>;
}

export class WikipediaToolOutput extends SearchToolOutput<WikipediaToolResult> {
  constructor(
    public readonly results: WikipediaToolResult[],
    protected readonly maxSerializedLength: number,
  ) {
    super(results);
  }

  static {
    this.register();
  }

  @Cache()
  getTextContent(): string {
    if (this.isEmpty()) {
      return `No results were found. Try to reformat your query.`;
    }

    const target = this.results.length === 1 ? this.results[0] : this.results;
    const response = JSON.stringify(target);

    return this.maxSerializedLength < Infinity
      ? response.substring(0, this.maxSerializedLength)
      : response;
  }

  createSnapshot() {
    return {
      results: this.results,
      maxSerializedLength: this.maxSerializedLength,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

interface SearchResponse {
  results: Pick<pageResult, "title" | "pageid">[];
  suggestion: string;
}

export class WikipediaTool extends Tool<
  WikipediaToolOutput,
  WikipediaToolOptions,
  WikipediaToolRunOptions
> {
  name = "Wikipedia";
  description =
    "Search factual and historical information, including biography, history, politics, geography, society, culture, science, technology, people, animal species, mathematics, and other subjects.";

  public readonly emitter: ToolEmitter<ToolInput<this>, WikipediaToolOutput> = Emitter.root.child({
    namespace: ["tool", "search", "wikipedia"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      query: z
        .string({ description: `Name of the wikipedia page, for example 'New York'` })
        .min(1)
        .max(128),
    });
  }

  public constructor(public readonly config: WikipediaToolOptions = {}) {
    super(config);
  }

  static {
    this.register();
  }

  @Cache({
    enumerable: false,
  })
  protected get _mappers(): Record<
    keyof PageFunctions,
    (page: Page, runOptions: WikipediaToolRunOptions) => Promise<any>
  > {
    return {
      categories: (page) => page.categories(),
      content: (page) => page.content(),
      html: (page) => page.html(),
      markdown: async (page, runOptions) => {
        const html = await page.html().then((result) => {
          const url = new URL(page.fullurl);
          const base = `${url.protocol}//${[url.hostname, url.port].filter(Boolean).join(":")}`;
          return (
            result
              // Missing a protocol
              .replace(/(<img .*src=)"(\/\/.*)"/gm, `$1"${url.protocol}$2"`)
              .replace(/(<a .*href=)"(\/\/.*)"/gm, `$1"${url.protocol}$2"`)

              // Missing a hostname
              .replace(/(<img .*src=)"(\/.*)"/gm, `$1"${base}$2"`)
              .replace(/(<a .*href=)"(\/.*)"/gm, `$1"${base}$2"`)
          );
        });

        const service = new Turndown();
        service.use(turndownPlugin.gfm);
        return service
          .remove((node) => runOptions.extraction?.fields?.markdown?.filter?.(node) === false)
          .turndown(html);
      },
      images: (page) => page.images(),
      intro: (page) => page.intro(),
      infobox: (page) => page.infobox(),
      links: (page) => page.links(),
      coordinates: (page) => page.coordinates(),
      langLinks: (page) => page.langLinks(),
      references: (page) => page.references(),
      related: (page) => page.related(),
      summary: (page) => page.summary(),
      tables: (page) => page.tables(),
    };
  }

  @Cache({
    enumerable: false,
  })
  protected get _defaultRunOptions(): WikipediaToolRunOptions {
    const unwrapTags = new Set(["a", "small", "sup"]);
    const unwrapTagsSelector = Array.from(unwrapTags.values()).join(",");

    const ignoredTags = new Set([
      "img",
      "link",
      "style",
      "abbr",
      "cite",
      "input",
      "sup",
      "bdi",
      "q",
      "figure",
      "audio",
      "track",
      "figcaption",
    ]);
    const ignoredClasses = new Set([
      "toc",
      "reflist",
      "mw-references-wrap",
      "box-More_footnotes_needed",
      "navbox",
      "navbox-styles",
      "mw-editsection",
      "sistersitebox",
      "navbox-inner",
      "refbegin",
      "notpageimage",
      "mw-file-element",
      "box-Unreferenced_section",
      "navigation-not-searchable",
      "noexcerpt",
      "infobox-image",
      "mw-tmh-player",
      "printfooter",
      "ambox",
      "infobox",
    ]);
    const ignoredRoles = new Set(["navigation"]);

    const ignoredSelector = [
      ...ignoredTags.values(),
      Array.from(ignoredClasses.values()).map((cls) => `.${cls}`),
      Array.from(ignoredRoles.values()).map((role) => `[role="${role}"]`),
      "table>caption",
    ].join(",");

    return {
      extraction: {
        fields: {
          markdown: {
            transform: (output: string) => output.replace(/([^\\])(\$)/g, `$1\\$2`),
            filter: (node) => {
              const tagName = node.tagName.toLowerCase();
              if (ignoredTags.has(tagName) || ignoredRoles.has(node.role ?? "")) {
                return false;
              }

              for (const cls of node.className.trim().split(" ").filter(Boolean)) {
                if (ignoredClasses.has(cls)) {
                  return false;
                }
              }

              for (const table of Array.from(node.querySelectorAll("table tr>th"))) {
                let colspan = parseInt(table.getAttribute("colspan") || "1");
                while (colspan > 1) {
                  table.insertAdjacentHTML("afterend", `<th>&nbsp;</th>`);
                  colspan--;
                }
                table.removeAttribute("colspan");
              }

              const ignoredTargets = Array.from(
                ignoredSelector ? node.querySelectorAll(ignoredSelector) : [],
              );
              for (const childNode of ignoredTargets) {
                childNode.remove();
              }

              const unwrapTargets = Array.from(
                unwrapTagsSelector ? node.querySelectorAll(unwrapTagsSelector) : [],
              ).reverse();
              if (unwrapTags.has(tagName)) {
                unwrapTargets.push(node);
              }
              for (const childNode of unwrapTargets) {
                childNode.outerHTML = childNode.innerHTML;
              }

              return true;
            },
          },
        },
      },
      filters: {
        minPageNameSimilarity: 0.5,
        excludeOthersOnExactMatch: true,
      },
      search: {
        limit: 3,
        suggestion: true,
      },
      output: {
        maxSerializedLength: 25_000,
        maxDescriptionLength: 250,
      },
    };
  }

  protected _createRunOptions(overrides?: WikipediaToolRunOptions): WikipediaToolRunOptions {
    const baseKeys: ArrayKeys<Common<WikipediaToolRunOptions, WikipediaToolOptions>> = [
      "filters",
      "search",
      "extraction",
      "retryOptions",
      "output",
    ];

    return R.pipe(
      { ...this._defaultRunOptions },
      R.mergeDeep(R.pick(this.options ?? {}, baseKeys)),
      R.mergeDeep({ ...overrides }),
    );
  }

  protected async _run(
    { query: input }: ToolInput<WikipediaTool>,
    _options?: WikipediaToolRunOptions,
  ): Promise<WikipediaToolOutput> {
    const runOptions = this._createRunOptions(_options);

    const { results: searchRawResults, suggestion }: SearchResponse = await wiki.default.search(
      input,
      {
        suggestion: Boolean(_options?.search?.suggestion),
        ...runOptions.search,
      },
    );

    if (searchRawResults.length === 0 && suggestion && runOptions.search?.suggestion) {
      return await this._run({ query: suggestion }, _options);
    }

    const bestCandidates = this.extractBestCandidates(
      input,
      searchRawResults,
      runOptions?.filters ?? {},
    );

    const results = await Promise.all(
      bestCandidates.map(async ({ pageId }) => {
        // @ts-expect-error wrong library's typing, passing a string would lead to a classic text search instead of a concrete page retrieval
        const page = await wiki.default.page(pageId, {
          redirect: true,
          preload: false,
          fields: keys(runOptions.extraction?.fields ?? {}).filter((key) => key !== "markdown"),
        });

        return asyncProperties({
          title: page.title,
          description: ((): Promise<string> => {
            const length = runOptions?.output?.maxDescriptionLength ?? 0;
            return length <= 0
              ? Promise.resolve("")
              : page.content().then((content) => content.substring(0, length));
          })(),
          url: page.fullurl,
          fields: asyncProperties(
            mapValues(runOptions?.extraction?.fields ?? {}, (value, key) =>
              this._mappers[key](page, runOptions)
                .then((response) => (value.transform ? value.transform(response) : response))
                .catch(() => undefined),
            ),
          ),
        });
      }),
    );
    return new WikipediaToolOutput(results, runOptions.output?.maxSerializedLength ?? Infinity);
  }

  protected extractBestCandidates(
    query: string,
    candidates: SearchResponse["results"],
    options: FilterOptions,
  ) {
    const normalize = (text: string) =>
      text
        .normalize("NFKD")
        .replace(/[^\w| ]/g, "") // remove diacritics and special characters (except whitespace)
        .replace(/\s\s+/g, " ") // collapse multiple whitespaces into one
        .trim();

    const bestCandidates = stringComparison.jaccardIndex
      .sortMatch(
        normalize(query),
        candidates.map((candidate) => normalize(candidate.title)),
      )
      .map((result) => ({
        pageId: candidates[result.index].pageid,
        score: result.rating,
      }))
      .filter((result) => result.score >= (options.minPageNameSimilarity ?? 0))
      .sort((a, b) => b.score - a.score);

    if (bestCandidates.at(0)?.score === 1 && options.excludeOthersOnExactMatch) {
      bestCandidates.length = 1;
    }

    return bestCandidates;
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      config: this.config,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    super.loadSnapshot(snapshot);
  }
}
