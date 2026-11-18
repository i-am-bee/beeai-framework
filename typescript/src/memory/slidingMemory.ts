/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AssistantMessage, Message, ToolMessage } from "@/backend/message.js";
import { BaseMemory, MemoryFatalError } from "@/memory/base.js";
import { shallowCopy } from "@/serializer/utils.js";
import { filter, isTruthy } from "remeda";
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

// Assistant tool calls and their following results must leave the window together.
function toolExchange(messages: Message[], index: number): [number, number] {
  let start = index;
  while (
    start > 0 &&
    (messages[start] instanceof ToolMessage ||
      (messages[start] instanceof AssistantMessage &&
        messages[start - 1] instanceof AssistantMessage))
  ) {
    start--;
  }
  let end = start;
  while (end < messages.length && messages[end] instanceof AssistantMessage) {
    end++;
  }
  if (
    !messages
      .slice(start, end)
      .some((msg) => msg instanceof AssistantMessage && msg.getToolCalls().length > 0)
  ) {
    return [index, index + 1];
  }
  while (end < messages.length && messages[end] instanceof ToolMessage) {
    end++;
  }
  return index < end ? [start, end] : [index, index + 1];
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
      const remaining = this.messages.slice();
      const selected = filter(castArray(handlers.removalSelector(remaining)), isTruthy);
      const removed = new Set<Message>();
      const visited = new Set<Message>();
      const pendingIds = new Set(
        message instanceof ToolMessage
          ? message.getToolResults().map((result) => result.toolCallId)
          : [],
      );
      // only the latest call can own the incoming result; an older exchange may reuse its id
      const pendingCall = remaining
        .slice()
        .reverse()
        .find(
          (msg) =>
            msg instanceof AssistantMessage &&
            msg.getToolCalls().some((call) => pendingIds.has(call.toolCallId)),
        );

      for (const candidate of selected) {
        // A selector can include both a call and its result, already removed as a group.
        const seen = visited.has(candidate);
        visited.add(candidate);
        if (removed.has(candidate) && !seen) {
          continue;
        }
        const index = remaining.indexOf(candidate);
        if (index === -1) {
          throw new MemoryFatalError(`Cannot delete non existing message.`, [], {
            context: { message: candidate, messages: this.messages },
          });
        }
        const [start, end] = toolExchange(remaining, index);
        const exchange = remaining.slice(start, end);
        if (pendingCall && exchange.includes(pendingCall)) {
          throw new MemoryFatalError(
            `The tool exchange cannot fit inside current memory (${size} messages).`,
          );
        }
        remaining.splice(start, end - start);
        for (const msg of exchange) {
          removed.add(msg);
        }
      }

      if (remaining.length + 1 > size) {
        throw new MemoryFatalError(
          `Custom memory removalSelector did not return any message. Memory overflow has occurred.`,
        );
      }

      this.messages.length = 0;
      for (const stored of remaining) {
        this.messages.push(stored);
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
