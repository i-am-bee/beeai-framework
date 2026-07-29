/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createOpenAI, OpenAIProvider, OpenAIProviderSettings } from "@ai-sdk/openai";
import { getEnv } from "@/internals/env.js";
import { BackendClient } from "@/backend/client.js";
import { parseHeadersFromEnv, vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export const MINIMAX_API_BASE = "https://api.minimax.io/v1";
export const MINIMAX_API_BASE_CN = "https://api.minimaxi.com/v1";

/**
 * Region identifiers accepted by `region` / the MINIMAX_API_REGION env var,
 * mapped to the matching OpenAI-compatible base URL. MiniMax serves the same
 * API from two regional gateways; this mapping lets callers select one by name
 * instead of discovering and typing the base URL by hand.
 */
export const MINIMAX_REGION_BASE_URLS = {
  global: MINIMAX_API_BASE,
  global_en: MINIMAX_API_BASE,
  cn: MINIMAX_API_BASE_CN,
  cn_zh: MINIMAX_API_BASE_CN,
} as const;

export type MiniMaxRegion = keyof typeof MINIMAX_REGION_BASE_URLS;

/**
 * Resolve a MiniMax region identifier to its OpenAI-compatible base URL.
 *
 * An empty or missing region resolves to the global endpoint. Unknown
 * identifiers throw an error listing the supported regions.
 */
export function resolveMiniMaxBaseURL(region?: string): string {
  const normalized = region?.trim().toLowerCase();
  if (!normalized) {
    return MINIMAX_API_BASE;
  }
  const baseURL = (MINIMAX_REGION_BASE_URLS as Record<string, string>)[normalized];
  if (!baseURL) {
    throw new Error(
      `Unknown MiniMax region "${region}". Supported regions: ${Object.keys(
        MINIMAX_REGION_BASE_URLS,
      ).join(", ")}.`,
    );
  }
  return baseURL;
}

export interface MiniMaxClientSettings extends OpenAIProviderSettings {
  /**
   * Region whose endpoint is used when neither `baseURL` nor MINIMAX_API_BASE
   * is set. Falls back to the MINIMAX_API_REGION env var, then to the global
   * endpoint.
   */
  region?: MiniMaxRegion;
}

export class MiniMaxClient extends BackendClient<MiniMaxClientSettings, OpenAIProvider> {
  protected create(): OpenAIProvider {
    const { region, ...settings } = this.settings ?? {};
    return createOpenAI({
      ...settings,
      apiKey: settings.apiKey || getEnv("MINIMAX_API_KEY"),
      baseURL:
        settings.baseURL ||
        getEnv("MINIMAX_API_BASE") ||
        resolveMiniMaxBaseURL(region ?? getEnv("MINIMAX_API_REGION")),
      headers: {
        ...parseHeadersFromEnv("MINIMAX_API_HEADERS"),
        ...settings.headers,
      },
      fetch: vercelFetcher(settings.fetch),
    });
  }
}
