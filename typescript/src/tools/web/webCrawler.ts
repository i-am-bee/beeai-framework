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
  ToolInput,
} from "@/tools/base.js";
import { z } from "zod";
import { Cache } from "@/cache/decoratorCache.js";
import { stripHtml } from "string-strip-html";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";

interface CrawlerOutput {
  url: string;
  statusCode: number;
  statusText: string;
  contentType: string;
  content: string;
}

export class WebCrawlerToolOutput extends JSONToolOutput<CrawlerOutput> {
  @Cache()
  getTextContent(): string {
    return [
      `URL: ${this.result.url}`,
      `STATUS: ${this.result.statusCode} (${this.result.statusText})`,
      `CONTENT-TYPE: ${this.result.contentType}`,
      `CONTENT: ${this.result.content}`,
    ].join("\n");
  }
}

export type HttpClient = (url: string, options?: RequestInit) => Promise<HttpClientResponse>;
interface HttpClientResponse {
  status: number;
  statusText: string;
  headers: Headers;
  text(): Promise<string>;
}

type Parser = (response: HttpClientResponse) => Promise<string>;
interface WebsiteCrawlerToolOptions extends BaseToolOptions {
  client?: HttpClient;
  parser?: Parser;
  request?: RequestInit;
}

async function defaultParser(response: HttpClientResponse) {
  const text = await response.text();
  if (text) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("text/html")) {
      return stripHtml(text).result;
    }
  }
  return text || "No Content";
}

export class WebCrawlerTool extends Tool<WebCrawlerToolOutput, WebsiteCrawlerToolOptions> {
  name = "WebCrawler";
  description = `Retrieves content of an arbitrary website.`;
  inputSchema() {
    return z.object({
      url: z.string().url().describe("Website URL"),
    });
  }

  protected client: HttpClient;
  protected parser: Parser;

  public readonly emitter: ToolEmitter<ToolInput<this>, WebCrawlerToolOutput> = Emitter.root.child({
    namespace: ["tool", "webCrawler"],
    creator: this,
  });

  constructor({ client, parser, ...options }: WebsiteCrawlerToolOptions = {}) {
    super(options);
    this.client = client ?? fetch;
    this.parser = parser ?? defaultParser;
  }

  protected async _run(
    { url }: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ) {
    const response = await this.client(url, {
      redirect: "follow",
      ...this.options.request,
      signal: run.signal,
    });

    const content = await this.parser(response);
    return new WebCrawlerToolOutput({
      url,
      statusCode: response.status,
      statusText: response.statusText,
      contentType: response.headers.get("content-type") ?? "unknown",
      content,
    });
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      client: this.client,
      parser: this.parser,
    };
  }

  loadSnapshot({ client, parser, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    super.loadSnapshot(snapshot);
    Object.assign(this, {
      client: client ?? fetch,
      parser: parser ?? defaultParser,
    });
  }
}
