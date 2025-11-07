/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseMemory } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { removeFromArray } from "@/internals/helpers/array.js";
import { ensureRange } from "@/internals/helpers/number.js";
import { Message } from "@/backend/message.js";

export class UnconstrainedMemory extends BaseMemory {
  public messages: Message[] = [];

  static {
    this.register();
  }

  async add(message: Message, index?: number) {
    index = ensureRange(index ?? this.messages.length, { min: 0, max: this.messages.length });
    this.messages.splice(index, 0, message);
  }

  async delete(message: Message) {
    return removeFromArray(this.messages, message);
  }

  reset() {
    this.messages.length = 0;
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }

  createSnapshot() {
    return {
      messages: shallowCopy(this.messages),
    };
  }
}
