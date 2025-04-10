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

import { BaseMemory, MemoryFatalError } from "@/memory/base.js";
import * as R from "remeda";
import { shallowCopy } from "@/serializer/utils.js";
import { removeFromArray } from "@/internals/helpers/array.js";
import { map, sum } from "remeda";
import { ensureRange } from "@/internals/helpers/number.js";
import { Message } from "@/backend/message.js";

export interface Handlers {
  estimate: (messages: Message) => number;
  tokenize: (messages: Message[]) => Promise<number>;
  removalSelector: (messages: Message[]) => Message;
}

const simpleEstimate: Handlers["estimate"] = (msg: Message) => Math.ceil(msg.text.length / 4);
const simpleTokenize: Handlers["tokenize"] = async (msgs: Message[]) =>
  sum(map(msgs, simpleEstimate)); // TODO

export interface TokenMemoryInput {
  maxTokens?: number;
  syncThreshold?: number;
  capacityThreshold?: number;
  handlers?: Partial<Handlers>;
}

interface TokenByMessage {
  tokensCount: number;
  dirty: boolean;
}

export class TokenMemory extends BaseMemory {
  public readonly messages: Message[] = [];

  protected threshold;
  protected syncThreshold;
  protected maxTokens: number | null = null;
  protected tokensByMessage = new WeakMap<Message, TokenByMessage>();
  public readonly handlers: Handlers;

  constructor(config: TokenMemoryInput = {}) {
    super();
    this.maxTokens = config.maxTokens ?? null;
    this.threshold = config.capacityThreshold ?? 0.75;
    this.syncThreshold = config.syncThreshold ?? 0.25;
    this.handlers = {
      ...config?.handlers,
      estimate:
        config?.handlers?.estimate || ((msg) => Math.ceil((msg.role.length + msg.text.length) / 4)),
      tokenize: config?.handlers?.tokenize || simpleTokenize,
      removalSelector: config.handlers?.removalSelector || ((messages) => messages[0]),
    };
    if (!R.clamp({ min: 0, max: 1 })(this.threshold)) {
      throw new TypeError('"capacityThreshold" must be a number in range (0, 1>');
    }
  }

  static {
    this.register();
  }

  get tokensUsed(): number {
    return sum(this.messages.map((msg) => this.tokensByMessage.get(msg)!.tokensCount!));
  }

  get isDirty(): boolean {
    return this.messages.some((msg) => this.tokensByMessage.get(msg)?.dirty !== false);
  }

  async add(message: Message, index?: number) {
    if (this.maxTokens === null) {
      // TODO: improve
      this.maxTokens = 128_000;
    }

    const meta = this.tokensByMessage.has(message)
      ? this.tokensByMessage.get(message)!
      : { tokensCount: this.handlers.estimate(message), dirty: true };

    if (meta.tokensCount > this.maxTokens) {
      throw new MemoryFatalError(
        `Retrieved message (${meta.tokensCount} tokens) cannot fit inside current memory (${this.maxTokens} tokens)`,
      );
    }

    while (this.tokensUsed > this.maxTokens - meta.tokensCount) {
      const messageToDelete = this.handlers.removalSelector(this.messages);
      const exists = await this.delete(messageToDelete);

      if (!messageToDelete || !exists) {
        throw new MemoryFatalError('The "removalSelector" handler must return a valid message!');
      }
    }

    this.tokensByMessage.set(message, meta);

    index = ensureRange(index ?? this.messages.length, { min: 0, max: this.messages.length });
    this.messages.splice(index, 0, message);

    if (this.isDirty && this.tokensUsed / this.maxTokens >= this.syncThreshold) {
      await this.sync();
    }
  }

  async delete(message: Message) {
    return removeFromArray(this.messages, message);
  }

  async sync() {
    const messages = await Promise.all(
      this.messages.map(async (msg) => {
        const cache = this.tokensByMessage.get(msg);
        if (cache?.dirty !== false) {
          const tokensCount = await this.handlers.tokenize([msg]);
          this.tokensByMessage.set(msg, { tokensCount, dirty: false });
        }
        return msg;
      }),
    );

    this.messages.length = 0;
    await this.addMany(messages);
  }

  reset() {
    for (const msg of this.messages) {
      this.tokensByMessage.delete(msg);
    }
    this.messages.length = 0;
  }

  stats() {
    return {
      tokensUsed: this.tokensUsed,
      maxTokens: this.maxTokens,
      messagesCount: this.messages.length,
      isDirty: this.isDirty,
    };
  }

  createSnapshot() {
    return {
      threshold: this.threshold,
      syncThreshold: this.syncThreshold,
      messages: shallowCopy(this.messages),
      handlers: shallowCopy(this.handlers),
      maxTokens: this.maxTokens,
      tokensByMessage: this.messages
        .map((message) => [message, this.tokensByMessage.get(message)])
        .filter(([_, value]) => value !== undefined) as [Message, number][],
    };
  }

  loadSnapshot({ tokensByMessage, ...state }: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, state, {
      tokensByMessage: new WeakMap(tokensByMessage),
    });
  }
}
