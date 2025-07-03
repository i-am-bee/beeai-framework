/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ChatModel,
  ChatModelEvents,
  ChatModelOutput,
  ChatModelInput,
  ChatModelParameters,
} from "@/backend/chat.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { NotImplementedError } from "@/errors.js";

export class DummyChatModel extends ChatModel {
  public readonly emitter = Emitter.root.child<ChatModelEvents>({
    namespace: ["backend", "dummy", "chat"],
    creator: this,
  });

  constructor(
    public readonly modelId = "dummy",
    public readonly parameters: ChatModelParameters = {},
  ) {
    super();
  }

  get providerId(): string {
    return "dummy";
  }

  protected _create(_input: ChatModelInput, _run: GetRunContext<this>): Promise<ChatModelOutput> {
    throw new NotImplementedError();
  }

  protected _createStream(
    _input: ChatModelInput,
    _run: GetRunContext<this>,
  ): AsyncGenerator<ChatModelOutput> {
    throw new NotImplementedError();
  }

  createSnapshot() {
    return { ...super.createSnapshot(), modelId: this.modelId };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }

  static {
    this.register();
  }
}
