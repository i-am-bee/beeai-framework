/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LangChainEmbeddingModel } from "@/adapters/langchain/backend/embedding.js";
import { Embeddings } from "@langchain/core/embeddings";

class FakeEmbedding extends Embeddings {
  constructor(public readonly model?: string) {
    super({});
  }
  async embedDocuments(texts: string[]) {
    return texts.map(() => [0]);
  }
  async embedQuery() {
    return [0];
  }
}

class LegacyFakeEmbedding extends FakeEmbedding {
  modelName = "legacy-model";
}

describe("LangChainEmbeddingModel", () => {
  it("returns model from 'model' property", () => {
    const instance = new LangChainEmbeddingModel(new FakeEmbedding("text-embedding-ada-002"));
    expect(instance.modelId).toBe("text-embedding-ada-002");
  });

  it("falls back to modelName when model is absent", () => {
    const instance = new LangChainEmbeddingModel(new LegacyFakeEmbedding());
    expect(instance.modelId).toBe("legacy-model");
  });

  it("falls back to 'langchain' when neither property exists", () => {
    const instance = new LangChainEmbeddingModel(new FakeEmbedding());
    expect(instance.modelId).toBe("langchain");
  });

  it("returns 'langchain' as providerId", () => {
    const instance = new LangChainEmbeddingModel(new FakeEmbedding("any"));
    expect(instance.providerId).toBe("langchain");
  });
});
