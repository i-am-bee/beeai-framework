/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Message } from "@/backend/message.js";
import { BaseMemory, MemoryFatalError } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { filter, forEach, isTruthy, pipe } from "remeda";
import { castArray, removeFromArray } from "@/internals/helpers/array.js";
import { RequiredNested } from "@/internals/types.js";
import { ensureRange } from "@/internals/helpers/number.js";

export interface Handlers {
  removalSelector: (messages: Message[]) => Message | Message[];
}

export interface SlidingWindowMemoryInput {
  size: number;
  handlers?: Partial<Handlers>;
}

export class SlidingMemory extends BaseMemory {
  public readonly messages: Message[] = [];
  public readonly config: RequiredNested<SlidingWindowMemoryInput>;

  constructor(config: SlidingWindowMemoryInput) {
    super();
    this.config = {
      ...config,
      handlers: {
        removalSelector:
          config.handlers?.removalSelector ?? ((messages: Message[]) => [messages[0]]),
      },
    };
  }

  static {
    const aliases = ["SlidingWindowMemory"];
    this.register(aliases);
  }

  async add(message: Message, index?: number) {
    const { size, handlers } = this.config;
    const isOverflow = () => this.messages.length + 1 > size;

    if (isOverflow()) {
      pipe(
        this.messages,
        handlers.removalSelector,
        castArray,
        filter(isTruthy),
        forEach((message) => {
          const index = this.messages.indexOf(message);
          if (index === -1) {
            throw new MemoryFatalError(`Cannot delete non existing message.`, [], {
              context: { message, messages: this.messages },
            });
          }
          this.messages.splice(index, 1);
        }),
      );

      if (isOverflow()) {
        throw new MemoryFatalError(
          `Custom memory removalSelector did not return any message. Memory overflow has occurred.`,
        );
      }
    }

    index = ensureRange(index ?? this.messages.length, { min: 0, max: this.messages.length });
    this.messages.splice(index, 0, message);
  }

  async delete(message: Message) {
    return removeFromArray(this.messages, message);
  }

  reset() {
    this.messages.length = 0;
  }

  createSnapshot() {
    return { config: shallowCopy(this.config), messages: shallowCopy(this.messages) };
  }

  loadSnapshot(state: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, state);
  }
}
