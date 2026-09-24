/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
