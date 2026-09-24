/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { getEnv } from "@/internals/env.js";
import {
  createVertex,
  GoogleVertexProvider,
  GoogleVertexProviderSettings,
} from "@ai-sdk/google-vertex";
import { BackendClient } from "@/backend/client.js";
import { vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type GoogleVertexClientSettings = GoogleVertexProviderSettings;

export class GoogleVertexClient extends BackendClient<
  GoogleVertexClientSettings,
  GoogleVertexProvider
> {
  protected create(): GoogleVertexProvider {
    return createVertex({
      ...this.settings,
      project: this.settings?.project || getEnv("GOOGLE_VERTEX_PROJECT"),
      baseURL: this.settings?.baseURL || getEnv("GOOGLE_VERTEX_ENDPOINT"),
      location: this.settings?.location || getEnv("GOOGLE_VERTEX_LOCATION"),
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
