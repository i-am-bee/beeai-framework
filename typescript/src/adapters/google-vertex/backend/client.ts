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
