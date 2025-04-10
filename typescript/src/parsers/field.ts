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

import { ZodSchema } from "zod";
import { setProp } from "@/internals/helpers/object.js";
import { ValueError } from "@/errors.js";
import { Serializable } from "@/internals/serializable.js";
import { JSONParser } from "@streamparser/json";
import { jsonrepairTransform } from "jsonrepair/stream";
import { Cache, SingletonCacheKeyFn } from "@/cache/decoratorCache.js";
import { shallowCopy } from "@/serializer/utils.js";
import { parseBrokenJson } from "@/internals/helpers/schema.js";
import { findFirstPair } from "@/internals/helpers/string.js";

export abstract class ParserField<T, TPartial> extends Serializable {
  public raw = "";

  abstract get(): T;
  abstract getPartial(): TPartial;

  write(chunk: string) {
    this.raw += chunk;
  }

  async end() {}

  createSnapshot() {
    return { raw: this.raw };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

export class ZodParserField<T> extends ParserField<T, string> {
  static {
    this.register();
  }

  constructor(protected readonly schema: ZodSchema<T>) {
    super();
  }

  get(): T {
    return this.schema.parse(this.raw);
  }

  getPartial(): string {
    return this.raw;
  }

  createSnapshot() {
    return { ...super.createSnapshot(), schema: this.schema };
  }
}

export class JSONParserField<T> extends ParserField<T, Partial<T>> {
  protected stream!: ReturnType<typeof jsonrepairTransform>;
  protected jsonParser!: JSONParser;
  protected errored = false;
  protected ref!: { value: Partial<T> };

  constructor(
    protected readonly input: {
      schema: ZodSchema<T>;
      base: Partial<T>;
      matchPair?: [string, string];
    },
  ) {
    super();
    if (input.base === undefined) {
      throw new ValueError(`Base must be defined!`);
    }
    this.init();
  }

  @Cache({ cacheKey: SingletonCacheKeyFn })
  protected init() {
    this.ref = { value: shallowCopy(this.input.base) };
    this.jsonParser = new JSONParser({ emitPartialTokens: false, emitPartialValues: true });
    this.stream = jsonrepairTransform();
    this.stream.on("data", (chunk) => {
      if (this.errored) {
        return;
      }

      try {
        this.jsonParser.write(chunk.toString());
      } catch {
        this.errored = true;
      }
    });
    this.jsonParser.onValue = ({ value, key, stack }) => {
      const keys = stack
        .map((s) => s.key)
        .concat(key)
        .filter((s) => s !== undefined)
        .map(String);

      if (keys.length === 0 && value === undefined) {
        return;
      }
      const prefix: keyof typeof this.ref = "value";
      setProp(this.ref, [prefix, ...keys], value);
    };
  }

  write(chunk: string) {
    if (this.input.matchPair) {
      if (!this.raw) {
        const startChar = this.input.matchPair[0];
        const index = chunk.indexOf(startChar);
        if (index === -1) {
          return;
        }
        chunk = chunk.substring(index);
      } else {
        const merged = this.raw.concat(chunk);
        const match = findFirstPair(merged, this.input.matchPair);
        if (match) {
          if (match.end < this.raw.length) {
            return;
          }
          chunk = merged.substring(this.raw.length, match.end);
        }
      }
    }

    super.write(chunk);
    try {
      this.stream.push(chunk);
    } catch {
      this.errored = true;
    }
  }

  get() {
    const inputToParse = this.errored
      ? parseBrokenJson(this.raw, {
          pair: this.input.matchPair,
        })
      : this.ref.value;

    return this.input.schema.parse(inputToParse);
  }

  getPartial() {
    return this.ref.value;
  }

  async end() {
    if (this.stream.closed || this.jsonParser.isEnded || this.errored) {
      return;
    }

    return new Promise<void>((resolve, reject) => {
      this.jsonParser.onEnd = resolve;
      this.jsonParser.onError = reject;

      this.stream.push(null);
      this.jsonParser.end();
    });
  }

  createSnapshot() {
    return { ...super.createSnapshot(), input: this.input, errored: this.errored };
  }

  loadSnapshot({ raw, ...snapshot }: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, { raw: "", ...snapshot });
    this.init();
    this.write(raw);
  }
}

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace ParserField {
  export type inferValue<T> = T extends ParserField<infer L, unknown> ? L : never;
  export type inferPartialValue<T> = T extends ParserField<any, infer L> ? L : never;
}
