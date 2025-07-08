/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleVertexProvider } from "@ai-sdk/google-vertex";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import {
  GoogleVertexClient,
  GoogleVertexClientSettings,
} from "@/adapters/google-vertex/backend/client.js";
import { getEnv } from "@/internals/env.js";

type GoogleVertexParameters = Parameters<GoogleVertexProvider["languageModel"]>;
export type GoogleVertexChatModelId = NonNullable<GoogleVertexParameters[0]>;
export type GoogleVertexChatModelSettings = NonNullable<GoogleVertexParameters[1]>;

export class GoogleVertexChatModel extends VercelChatModel {
  constructor(
    modelId: GoogleVertexChatModelId = getEnv("GOOGLE_VERTEX_CHAT_MODEL", "gemini-1.5-pro"),
    settings: GoogleVertexChatModelSettings = {},
    client?: GoogleVertexClientSettings | GoogleVertexClient,
  ) {
    const model = GoogleVertexClient.ensure(client).instance.languageModel(modelId, settings);
    super(model);
  }
}
