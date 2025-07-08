/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { NotImplementedError } from "@/errors.js";
import {
  EmbeddingModel,
  EmbeddingModelEvents,
  EmbeddingModelInput,
  EmbeddingModelOutput,
} from "@/backend/embedding.js";

export class DummyEmbeddingModel extends EmbeddingModel {
  public readonly emitter = Emitter.root.child<EmbeddingModelEvents>({
    namespace: ["backend", "dummy", "embedding"],
    creator: this,
  });

  constructor(public readonly modelId = "dummy") {
    super();
  }

  get providerId(): string {
    return "dummy";
  }

  protected _create(
    _input: EmbeddingModelInput,
    _run: GetRunContext<this>,
  ): Promise<EmbeddingModelOutput> {
    throw new NotImplementedError();
  }

  createSnapshot() {
    return { ...super.createSnapshot(), modelId: this.modelId };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }
}
