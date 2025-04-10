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
