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
  createAmazonBedrock,
  AmazonBedrockProviderSettings,
  AmazonBedrockProvider,
} from "@ai-sdk/amazon-bedrock";
import { BackendClient } from "@/backend/client.js";
import { vercelFetcher } from "@/adapters/vercel/backend/utils.js";

export type AmazonBedrockClientSettings = AmazonBedrockProviderSettings;

export class AmazonBedrockClient extends BackendClient<
  AmazonBedrockClientSettings,
  AmazonBedrockProvider
> {
  protected create(): AmazonBedrockProvider {
    return createAmazonBedrock({
      ...this.settings,
      fetch: vercelFetcher(this.settings?.fetch),
    });
  }
}
