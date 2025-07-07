/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { WatsonxClient, WatsonxClientSettings } from "@/adapters/watsonx/backend/client.js";
import {
  EmbeddingModel,
  EmbeddingModelInput,
  EmbeddingModelOutput,
  EmbeddingModelEvents,
} from "@/backend/embedding.js";
import { EmbeddingParameters as WXEmbeddingParameters } from "@ibm-cloud/watsonx-ai/dist/watsonx-ai-ml/vml_v1.js";
import { Emitter } from "@/emitter/emitter.js";
import { getEnv } from "@/internals/env.js";

export type WatsonxEmbeddingModelParameters = WXEmbeddingParameters;

export class WatsonxEmbeddingModel extends EmbeddingModel {
  protected readonly client: WatsonxClient;
  public readonly emitter: Emitter<EmbeddingModelEvents>;

  get providerId() {
    return "watsonx";
  }

  constructor(
    public readonly modelId: string = getEnv(
      "WATSONX_EMBEDDING_MODEL",
      "ibm/granite-embedding-107m-multilingual",
    ),
    public readonly parameters: WatsonxEmbeddingModelParameters = {},
    client?: WatsonxClient | WatsonxClientSettings,
  ) {
    super();
    this.client = WatsonxClient.ensure(client);
    this.emitter = Emitter.root.child({
      namespace: ["backend", "watsonx", "embedding"],
      creator: this,
    });
  }

  protected async _create(input: EmbeddingModelInput): Promise<EmbeddingModelOutput> {
    const response = await this.client.instance.embedText({
      modelId: this.modelId,
      spaceId: this.client.spaceId,
      projectId: this.client.projectId,
      inputs: input.values,
      parameters: {
        return_options: { input_text: true },
        ...this.parameters,
      },
    });

    const embeddings = response.result.results.map((e) => e.embedding);
    const values = response.result.results.map((e, i) => e.input || input.values.at(i)!);

    return {
      embeddings,
      values,
      usage: {
        tokens: response.result.input_token_count,
      },
    };
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      modelId: this.modelId,
      client: this.client,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }
}
