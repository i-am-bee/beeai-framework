/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { getEnv } from "beeai-framework/internals/env";
import "dotenv/config";
import { State } from "./state.js";
import { Steps } from "./workflow.js";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

export function getChatLLM() {
  return new OllamaChatModel(getEnv("OLLAMA_CHAT_MODEL") || "deepseek-r1:8b");
}

export interface SearchResult {
  url: string;
  title: string;
  content: string;
  raw_content?: string | null;
}

export async function tavilySearch(query: string, maxResults = 3): Promise<SearchResult[]> {
  const apiKey = getEnv("TAVILY_API_KEY");
  if (!apiKey) {
    throw new Error("TAVILY_API_KEY not found in environment");
  }

  const tool = new TavilySearchResults({ apiKey, maxResults });
  const response = await tool.invoke(query);
  const parsed = JSON.parse(response);
  return parsed;
}

export function deduplicateAndFormatSources(
  searchResults: SearchResult[],
  maxTokensPerSource: number,
  includeRawContent = false,
): string {
  const uniqueSources = new Map<string, SearchResult>();

  searchResults.forEach((result) => {
    if (!uniqueSources.has(result.url)) {
      uniqueSources.set(result.url, result);
    }
  });

  const formattedParts: string[] = ["Sources:\n"];

  Array.from(uniqueSources.values()).forEach((source) => {
    formattedParts.push(`Source ${source.title}:\n===`);
    formattedParts.push(`URL: ${source.url}\n===`);
    formattedParts.push(`Most relevant content from source: ${source.content}\n===`);

    if (includeRawContent && source.raw_content) {
      const charLimit = maxTokensPerSource * 4;
      let rawContent = source.raw_content;

      if (rawContent.length > charLimit) {
        rawContent = rawContent.slice(0, charLimit) + "... [truncated]";
      }

      formattedParts.push(
        `Full source content limited to ${maxTokensPerSource} tokens: ${rawContent}\n`,
      );
    }
  });

  return formattedParts.join("\n").trim();
}

export function formatSources(results: SearchResult[]): string {
  return results.map((source) => `* ${source.title} : ${source.url}`).join("\n");
}

export enum RelevantOutputType {
  START = "start",
  FINISH = "finish",
  ERROR = "error",
}

export function getRelevantOutput(type: RelevantOutputType, step: Steps, state: State): string {
  if (type === RelevantOutputType.ERROR) {
    return "❌ An error occurred during execution.";
  }

  switch (step) {
    case Steps.GENERATE_COMPETITORS: {
      return type === RelevantOutputType.START
        ? `Analyzing ${state.industry} industry...`
        : `Found competitors: ${state.competitors.join(", ")}`;
    }
    case Steps.SELECT_COMPETITOR: {
      return type === RelevantOutputType.START
        ? "Selecting next competitor..."
        : `Analyzing: ${state.currentCompetitor}`;
    }
    case Steps.WEB_RESEARCH: {
      return type === RelevantOutputType.START
        ? `🔎 Researching: "${state.searchQuery}"`
        : "Research complete";
    }
    case Steps.CATEGORIZE_FINDINGS: {
      return type === RelevantOutputType.START
        ? "Categorizing findings..."
        : "Categorization complete";
    }
    case Steps.FINALIZE_SUMMARY: {
      return type === RelevantOutputType.START
        ? "Generating final analysis..."
        : "Analysis complete";
    }
    default: {
      return "";
    }
  }
}
