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

import { Mock } from "vitest";
import * as R from "remeda";
import { Serializer } from "@/serializer/serializer.js";
import { ClassConstructor } from "@/internals/types.js";
import { Logger } from "@/logger/logger.js";
import { pino } from "pino";
import { BaseLLM as LCBaseLLM } from "@langchain/core/language_models/llms";
import { ZodType } from "zod";
import { Callback } from "@/emitter/types.js";
import { RunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { toJsonSchema } from "@/internals/helpers/schema.js";
import { customsearch_v1 } from "@googleapis/customsearch";
import { LangChainTool } from "@/adapters/langchain/tools.js";
import { Client as esClient } from "@elastic/elasticsearch";
import { VercelChatModel } from "@/adapters/vercel/backend/chat.js";
import { isWeakKey } from "@/internals/helpers/weakRef.js";

interface CallbackOptions<T> {
  required?: boolean;
  check?: Callback<T>;
}

export function createCallbackRegister() {
  return {
    _container: new Map<string, { fn: Mock; options: CallbackOptions<any> }>(),
    create<T>(name: string, options: CallbackOptions<T> = {}) {
      const fn = vi.fn();
      if (options.check) {
        fn.mockImplementation(options?.check);
      }
      if (this._container.has(name)) {
        throw new Error(`Function '${name}' already registered!`);
      }
      this._container.set(name, {
        fn,
        options: { required: true, ...R.pickBy(R.isDefined)(options) },
      });
      return fn as NonNullable<typeof options.check>;
    },
    [Symbol.iterator]() {
      return this._container.entries();
    },
    verify(handler?: (ctx: { name: string; fn: Mock; options: CallbackOptions<any> }) => void) {
      for (const [name, { fn, options }] of this._container.entries()) {
        if (options.required) {
          expect(fn).toHaveBeenCalled();
        }
        handler?.({ name, fn, options });
      }
    },
  };
}

export function verifyDeserialization(
  ref: unknown,
  deserialized: unknown,
  parent?: any,
  path: string[] = [],
  ignoredPaths: string[] = [],
  seen = new WeakSet(),
) {
  if (isWeakKey(ref)) {
    if (seen.has(ref)) {
      return;
    }
    seen.add(ref);
  }

  if (R.isPromise(ref) || R.isPromise(deserialized)) {
    throw new TypeError('Value passed to "verifyDeserialization" is promise (forgotten await)!');
  }

  if (R.isFunction(ref) && R.isFunction(deserialized)) {
    expect(deserialized.toString()).toStrictEqual(ref.toString());
    return;
  }

  if (R.isObjectType(ref) && R.isObjectType(deserialized)) {
    const getNonIgnoredKeys = (instance: any) =>
      new Set(
        Object.entries(instance)
          .filter(([key, value]) => !verifyDeserialization.isIgnored(key, value, instance))
          .map(([key, _]) => String(key))
          .sort(),
      );

    const refKeys = getNonIgnoredKeys(ref);
    const keysB = getNonIgnoredKeys(deserialized);
    expect(keysB).toStrictEqual(refKeys);

    for (const key of Array.from(refKeys.values())) {
      const fullPath = path.concat(key).join(".");
      if (ignoredPaths.includes(fullPath)) {
        continue;
      }

      let value: any = ref[key as keyof typeof ref];
      let target: any = deserialized[key as keyof typeof deserialized];

      if (value instanceof ZodType) {
        value = toJsonSchema(value);
      }
      if (target instanceof ZodType) {
        target = toJsonSchema(target);
      }
      Serializer.findFactory(target);
      verifyDeserialization(value, target, parent, path.concat(key), ignoredPaths, seen);
    }
  } else {
    expect(deserialized).toStrictEqual(ref);
  }
}
verifyDeserialization.ignoredKeys = new Set<string>([LangChainTool.serializedSchemaKey]);
verifyDeserialization.ignoredClasses = [
  Logger,
  customsearch_v1.Customsearch,
  LCBaseLLM,
  RunContext,
  Emitter,
  esClient,
] as ClassConstructor[];
verifyDeserialization.isIgnored = (key: string, value: unknown, parent?: any) => {
  if (verifyDeserialization.ignoredKeys.has(key)) {
    return true;
  }

  const ignored = verifyDeserialization.ignoredClasses;

  // Pino check
  if (R.isObjectType(value) && Object.values(pino.symbols).some((symbol) => symbol in value)) {
    return true;
  }

  if (parent && parent instanceof VercelChatModel) {
    try {
      Serializer.findFactory(value);
      return false;
    } catch {
      return true;
    }
  }

  return ignored.some((Class) => value instanceof Class);
};
