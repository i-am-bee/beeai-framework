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

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  JSONToolOutput,
  Tool,
  ToolInput,
} from "./base.js";
import { string, z } from "zod";
import { RunContext } from "@/context.js";
import { filter, map, pipe, prop, sortBy, take } from "remeda";
import { Emitter } from "@/emitter/emitter.js";

const documentSchema = z.object({ text: string() }).passthrough();

type Document = z.infer<typeof documentSchema>;

interface ProviderInput {
  query: string;
  documents: Document[];
}

type Provider<TProviderOptions> = (
  input: ProviderInput,
  options: TProviderOptions | undefined,
  run: RunContext<SimilarityTool<TProviderOptions>>,
) => Promise<{ score: number }[]>;

export interface SimilarityToolOptions<TProviderOptions = unknown> extends BaseToolOptions {
  provider: Provider<TProviderOptions>;
  maxResults?: number;
}

export interface SimilarityToolRunOptions<TProviderOptions = unknown> extends BaseToolRunOptions {
  provider?: TProviderOptions;
  maxResults?: number;
  minScore?: number;
}

export interface SimilarityToolResult {
  document: Document;
  index: number;
  score: number;
}

export class SimilarityToolOutput extends JSONToolOutput<SimilarityToolResult[]> {}

export class SimilarityTool<TProviderOptions> extends Tool<
  SimilarityToolOutput,
  SimilarityToolOptions<TProviderOptions>,
  SimilarityToolRunOptions<TProviderOptions>
> {
  name = "Similarity";
  description = "Extract relevant information from documents.";

  public readonly emitter: ToolEmitter<ToolInput<this>, SimilarityToolOutput> = Emitter.root.child({
    namespace: ["tool", "similarity"],
    creator: this,
  });

  inputSchema() {
    return z.object({ query: z.string(), documents: z.array(documentSchema) });
  }

  static {
    this.register();
  }

  protected async _run(
    { query, documents }: ToolInput<this>,
    options: Partial<SimilarityToolRunOptions<TProviderOptions>>,
    run: RunContext<this>,
  ) {
    return pipe(
      await this.options.provider(
        {
          query,
          documents,
        },
        options?.provider,
        run,
      ),
      map(({ score }, idx) => ({
        documentIndex: idx,
        score,
      })),
      filter(({ score }) => score >= (options.minScore ?? -Infinity)),
      sortBy([prop("score"), "desc"]),
      take(options?.maxResults ?? this.options.maxResults ?? Infinity),
      (data) =>
        new SimilarityToolOutput(
          data.map(({ documentIndex, score }) => ({
            document: documents[documentIndex],
            index: documentIndex,
            score,
          })),
        ),
    );
  }
}
