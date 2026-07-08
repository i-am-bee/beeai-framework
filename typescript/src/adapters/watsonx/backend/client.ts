/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { getEnv } from "@/internals/env.js";
import { IamAuthenticator, UserOptions } from "ibm-cloud-sdk-core";
import { BackendClient } from "@/backend/client.js";

export interface WatsonxClientSettings extends Pick<UserOptions, "authenticator" | "version"> {
  spaceId?: string;
  baseUrl?: string;
  region?: string;
  projectId?: string;
  apiKey?: string;
}

export class WatsonxClient extends BackendClient<WatsonxClientSettings, WatsonXAI> {
  constructor(settings: WatsonxClientSettings) {
    const region = settings?.region || getEnv("WATSONX_REGION");
    const baseUrl =
      settings?.baseUrl || getEnv("WATSONX_BASE_URL") || `https://${region}.ml.cloud.ibm.com`;

    const projectId = settings?.projectId || getEnv("WATSONX_PROJECT_ID");
    const spaceId = projectId ? undefined : settings?.spaceId || getEnv("WATSONX_SPACE_ID");
    const version = settings?.version || getEnv("WATSONX_VERSION") || "2024-05-31";

    super({
      ...settings,
      baseUrl,
      projectId,
      spaceId,
      version,
    });
  }
  get spaceId() {
    return this.settings.spaceId;
  }

  get projectId() {
    return this.settings.projectId;
  }

  protected create() {
    return WatsonXAI.newInstance({
      version: this.settings.version,
      serviceUrl: this.settings.baseUrl,
      authenticator:
        this.settings?.authenticator ||
        new IamAuthenticator({
          apikey: this.settings?.apiKey || getEnv("WATSONX_API_KEY", ""),
          url: "https://iam.cloud.ibm.com",
        }),
    });
  }
}
