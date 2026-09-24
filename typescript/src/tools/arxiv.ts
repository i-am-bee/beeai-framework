/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  JSONToolOutput,
  Tool,
  ToolError,
  ToolInput,
  ToolInputValidationError,
} from "@/tools/base.js";
import { z } from "zod";
import { createURLParams } from "@/internals/fetcher.js";
import { Cache } from "@/cache/decoratorCache.js";
import { XMLParser } from "fast-xml-parser";
import { getProp } from "@/internals/helpers/object.js";
import { isDefined, isEmpty, pickBy } from "remeda";
import { castArray } from "@/internals/helpers/array.js";
import { ValueOf } from "@/internals/types.js";
import { AnyToolSchemaLike } from "@/internals/helpers/schema.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";

type ToolOptions = BaseToolOptions;
type ToolRunOptions = BaseToolRunOptions;

export const SortType = {
  RELEVANCE: "relevance",
  LAST_UPDATED_DATE: "lastUpdatedDate",
  SUBMITTED_DATE: "submittedDate",
} as const;

export const SortOrder = {
  ASCENDING: "ascending",
  DESCENDING: "descending",
} as const;

export const FilterType = {
  ALL: "all",
  TITLE: "title",
  AUTHOR: "author",
  ABSTRACT: "abstract",
  COMMENT: "comment",
  JOURNAL_REFERENCE: "journal_reference",
  SUBJECT_CATEGORY: "subject_category",
  REPORT_NUMBER: "report_number",
} as const;

const FilterTypeMapping: Record<ValueOf<typeof FilterType>, string> = {
  all: "all",
  title: "ti",
  author: "au",
  abstract: "abs",
  comment: "co",
  journal_reference: "jr",
  subject_category: "cat",
  report_number: "rn",
};

export const Separators = {
  AND: "+AND+",
  OR: "+OR+",
  ANDNOT: "+ANDNOT+",
} as const;

export interface ArXivResponse {
  totalResults: number;
  startIndex: number;
  itemsPerPage: number;
  entries: {
    id: string;
    title: string;
    summary: string;
    published: string;
    updated: string;
    authors: { name: string; affiliation: string[] }[];
    doi: string;
    comment: string;
    journalReference: string;
    primaryCategory: string;
    categories: string[];
    links: string[];
  }[];
}

export class ArXivToolOutput extends JSONToolOutput<ArXivResponse> {
  isEmpty(): boolean {
    return !this.result || this.result.totalResults === 0 || this.result.entries.length === 0;
  }
}

const extractId = (value: string) =>
  value
    .replace("https://arxiv.org/abs/", "")
    .replace("https://arxiv.org/pdf/", "")
    .replace(/v\d$/, "");

export class ArXivTool extends Tool<ArXivToolOutput, ToolOptions, ToolRunOptions> {
  name = "ArXiv";
  description = `Retrieves research articles published on arXiv including related metadata.`;

  public readonly emitter: ToolEmitter<ToolInput<this>, ArXivToolOutput> = Emitter.root.child({
    namespace: ["tool", "search", "arxiv"],
    creator: this,
  });

  @Cache()
  inputSchema() {
    const entrySchema = z.object({
      field: z.nativeEnum(FilterType).default(FilterType.ALL),
      value: z.string().min(1),
    });

    return z
      .object({
        ids: z.array(z.string().min(1)).optional(),
        search_query: z
          .object({
            include: z.array(entrySchema).nonempty().describe("Filters to include results."),
            exclude: z.array(entrySchema).optional().describe("Filters to exclude results."),
          })
          .optional(),
        start: z.number().int().min(0).default(0),
        maxResults: z.number().int().min(1).max(100).default(5),
      })
      .describe("Sorting by date is not supported.");
  }

  static {
    this.register();
  }

  protected validateInput(
    schema: AnyToolSchemaLike,
    rawInput: unknown,
  ): asserts rawInput is ToolInput<this> {
    super.validateInput(schema, rawInput);
    if (isEmpty(rawInput.ids ?? []) && !rawInput.search_query) {
      throw new ToolInputValidationError(
        `The 'search_query' property must be non-empty if the 'ids' property is not provided!`,
      );
    }
  }

  protected _prepareParams(input: ToolInput<typeof this>) {
    return createURLParams({
      start: input.start,
      max_results: input.maxResults,
      id_list: isEmpty(input.ids ?? []) ? undefined : input.ids?.map(extractId),
      search_query:
        input.search_query &&
        [
          input.search_query.include
            .map((tag) => `${FilterTypeMapping[tag.field]}:${tag.value}`)
            .join(Separators.AND),
          (input.search_query.exclude ?? [])
            .map((tag) => `${FilterTypeMapping[tag.field]}:${tag.value}`)
            .join(Separators.ANDNOT),
        ]
          .filter(Boolean)
          .join(Separators.ANDNOT),
      sortBy: SortType.RELEVANCE,
      sortOrder: SortOrder.DESCENDING,
    });
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ) {
    const params = this._prepareParams(input);
    const url = `https://export.arxiv.org/api/query?${decodeURIComponent(params.toString())}`;
    const response = await fetch(url, {
      signal: run.signal,
    });
    const data = await this._parseResponse(response);
    return new ArXivToolOutput(data);
  }

  protected async _parseResponse(response: Response): Promise<ArXivResponse> {
    const parser = new XMLParser({
      allowBooleanAttributes: true,
      alwaysCreateTextNode: false,
      attributeNamePrefix: "@_",
      attributesGroupName: false,
      cdataPropName: "#cdata",
      ignoreAttributes: true,
      numberParseOptions: { hex: false, leadingZeros: true },
      parseAttributeValue: false,
      parseTagValue: true,
      preserveOrder: false,
      removeNSPrefix: true,
      textNodeName: "#text",
      trimValues: true,
      ignoreDeclaration: true,
    });

    const text = await response.text();
    const parsedData = parser.parse(text);

    if (!response.ok) {
      throw new ToolError("Request to ArXiv API has failed!", [
        new Error(JSON.stringify(getProp(parsedData, ["feed", "entry"], parsedData), null, 2)),
      ]);
    }

    let entries: Record<string, any>[] = getProp(parsedData, ["feed", "entry"], []);
    entries = castArray(entries);

    return {
      totalResults: Math.max(getProp(parsedData, ["feed", "totalResults"], 0), entries.length),
      startIndex: getProp(parsedData, ["feed", "startIndex"], 0),
      itemsPerPage: getProp(parsedData, ["feed", "itemsPerPage"], 0),
      entries: entries.map((entry) =>
        pickBy(
          {
            id: extractId(entry.id),
            url: entry.id,
            title: entry.title,
            summary: entry.summary,
            published: entry.published,
            updated: entry.updated,
            authors: castArray(entry.author)
              .filter(Boolean)
              .map((author: any) => ({
                name: author.name,
                affiliation: castArray(author.affiliation ?? []),
              })),
            doi: entry.doi,
            comment: entry.comment,
            journalReference: entry.journal_ref,
            primaryCategory: entry.primary_category,
            categories: castArray(entry.category).filter(Boolean),
            links: castArray(entry.link).filter(Boolean),
          },
          isDefined,
        ),
      ),
    };
  }
}
