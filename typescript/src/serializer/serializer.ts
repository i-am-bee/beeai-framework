/**
 * Copyright 2025 IBM Corp.
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

import * as R from "remeda";
import { Serializable, SerializableClass } from "@/internals/serializable.js";
import { AnyConstructable, ClassConstructor, NamedFunction } from "@/internals/types.js";
import { SafeWeakMap, SafeWeakSet } from "@/internals/helpers/weakRef.js";
import { deserializeError, serializeError } from "serialize-error-cjs";
import { Version } from "@/version.js";
import {
  extractClassName,
  getFunctionBinds,
  isRootNode,
  isSerializationRequired,
  isSerializerNode,
  primitiveToSerializableClass,
  RefPlaceholder,
  RootNode,
  SerializerNode,
  SerializerRefIdentifier,
  SerializerSelfRefIdentifier,
  toBoundedFunction,
  traverseObject,
  traverseWithUpdate,
} from "@/serializer/utils.js";
import { SlidingTaskMap, Task, TaskState } from "promise-based-task";
import { getProp, setProp } from "@/internals/helpers/object.js";
import { halveString } from "@/internals/helpers/string.js";
import { traversePrototypeChain } from "@/internals/helpers/prototype.js";
import { CacheFn } from "@/cache/decoratorCache.js";
import { SerializerError } from "@/serializer/error.js";
import { ZodType } from "zod";
import { toJsonSchema } from "@/internals/helpers/schema.js";
import { createAbortController } from "@/internals/helpers/cancellation.js";
import { hasMinLength } from "@/internals/helpers/array.js";

export interface SerializeFactory<A = unknown, B = unknown> {
  ref: ClassConstructor<A> | NamedFunction<A>;
  createEmpty?: () => A;
  updateInstance?: (instance: A, update: A) => void | Promise<void>;
  toPlain: (value: A) => B | Promise<B>;
  fromPlain: (value: B, ref: ClassConstructor<A> | NamedFunction<A>) => A | Promise<A>;
}

export class Serializer {
  private static factories = new Map<string, SerializeFactory<any, any>>();

  static registerSerializable<A>(
    ref: SerializableClass<A>,
    processors?: Partial<Omit<SerializeFactory<Serializable<A>, A>, "ref">>,
    aliases?: string[],
  ) {
    return Serializer.register(
      ref,
      {
        toPlain: (value) => value.createSnapshot(),
        fromPlain: (value) => ref.fromSnapshot(value),
        createEmpty: () => Object.create(ref.prototype),
        updateInstance: (instance, update) => instance.loadSnapshot(update),
        ...R.pickBy(processors ?? {}, R.isDefined),
      },
      aliases,
    );
  }

  static deregister(ref: ClassConstructor | NamedFunction) {
    const className = extractClassName(ref);
    Serializer.factories.delete(className);
  }

  static register<A, B = unknown>(
    ref: ClassConstructor<A> | NamedFunction<A>,
    processors: Omit<SerializeFactory<A, B>, "ref">,
    aliases?: string[],
  ): void {
    const className = extractClassName(ref);
    const oldFactory = Serializer.factories.get(className);
    const newFactory: SerializeFactory<A, B> = {
      ref,
      ...processors,
    };

    if (oldFactory) {
      if (oldFactory.ref !== ref) {
        throw new SerializerError(`Factory for class "${className}" already exists!`);
      }

      for (const [key, value] of Serializer.factories.entries()) {
        if (value === oldFactory) {
          Serializer.factories.set(key, newFactory);
        }
      }
    }
    Serializer.factories.set(className, newFactory);

    aliases?.forEach((alias) => {
      const aliasTarget = Serializer.factories.get(alias);
      if (!aliasTarget) {
        this.factories.set(alias, newFactory);
      } else if (aliasTarget !== newFactory) {
        throw new SerializerError(
          `Factory for class "${className}" already exists! Cannot add alias.`,
        );
      }
    });
  }

  static getFactory(clsName: string): SerializeFactory {
    const factory = Serializer.factories.get(clsName);
    if (!factory) {
      throw new SerializerError(`Class "${clsName}" was not found!`);
    }
    return factory;
  }

  static findFactory(value: unknown) {
    let targetClass = extractClassName(value);
    if (!Serializer.hasFactory(targetClass) && R.isFunction(value)) {
      targetClass = Function.name;
    }

    if (Serializer.hasFactory(targetClass)) {
      const factory = Serializer.getFactory(targetClass);
      return { targetClass, factory, isSelfRef: factory.ref === value };
    }

    const excluded = new Set([null, Object.prototype]);
    for (const proto of traversePrototypeChain(value, excluded)) {
      const targetClass = extractClassName(proto);

      const factory = this.factories.get(targetClass);
      if (factory) {
        return { targetClass, factory, isSelfRef: false };
      }
    }

    throw new SerializerError(`Class "${targetClass}" was not found!`);
  }

  static hasFactory(clsName: string) {
    return Serializer.factories.has(clsName);
  }

  static async serialize<T>(rawData: T): Promise<string> {
    const output = Serializer._createOutputBuilder();
    const getRefId = (() => {
      let id = 0;
      return () => {
        id += 1;
        return id;
      };
    })();

    const seen = new SafeWeakMap<SerializerNode>();

    const assertValidSnapshot = (() => {
      const allowedTypes = new Set([
        "Number",
        "String",
        "Object",
        "Array",
        "Undefined",
        "Null",
        "Boolean",
      ]);
      return (snapshot: unknown, factory: SerializeFactory) => {
        const className = extractClassName(snapshot);

        if (!allowedTypes.has(className)) {
          throw new SerializerError(
            `Cannot serialize '${className}' as a top level property. "toPlain" function in "${extractClassName(factory.ref)}" should return an ${Array.from(allowedTypes.values()).join(",")}.`,
          );
        }
      };
    })();

    const toSerializable = async (rawValue: unknown): Promise<SerializerNode | unknown> => {
      if (seen.has(rawValue)) {
        return seen.get(rawValue);
      }

      if (rawValue instanceof Serializable && !this.hasFactory(extractClassName(rawValue))) {
        const Class = rawValue.constructor as SerializableClass<T>;
        Serializer.registerSerializable(Class);
      }

      const { targetClass, factory, isSelfRef } = Serializer.findFactory(rawValue);
      if (!isSerializationRequired(factory.ref)) {
        return rawValue;
      }

      const snapshot = isSelfRef ? SerializerSelfRefIdentifier : await factory.toPlain(rawValue);
      assertValidSnapshot(snapshot, factory);

      const result: SerializerNode = {
        __serializer: true,
        __class: targetClass,
        __ref: getRefId().toString(),
        __value: snapshot,
      };
      seen.set(rawValue, result);

      for (const node of traverseWithUpdate(snapshot)) {
        const newValue = await toSerializable(node.value);
        if (newValue !== node.value) {
          node.update(newValue);
        }
      }
      return result;
    };

    const root: RootNode<T> = { __version: Version, __root: rawData };
    await traverseObject(root, async ({ value, path }) => {
      const content = await toSerializable(value);
      output.update(path, content);
    });
    return output.toJSON();
  }

  /** @internal */
  static async deserializeWithMeta<T = any>(
    raw: string,
    extraClasses?: SerializableClass<unknown>[],
  ): Promise<RootNode<T>> {
    extraClasses?.forEach((ref) => Serializer.registerSerializable(ref));

    const output = Serializer._createOutputBuilder<RootNode<T>>();
    const instances = new Map<string, unknown>();

    const toDeserialize = async (contentRaw: unknown) => {
      if (isSerializerNode(contentRaw)) {
        const clsName = String(contentRaw.__class);
        const factory = Serializer.getFactory(clsName);
        const rawData = contentRaw.__value;

        if (rawData === SerializerSelfRefIdentifier) {
          return factory.ref;
        }

        if (rawData === SerializerRefIdentifier) {
          if (!instances.has(contentRaw.__ref!)) {
            throw new SerializerError(`Missing reference "${contentRaw.__ref}"!`);
          }

          const data = instances.get(contentRaw.__ref!)!;
          if (data instanceof RefPlaceholder) {
            return data.value;
          }
          return data;
        }

        const traverseNested = async () => {
          for (const node of traverseWithUpdate(rawData)) {
            const newValue = await toDeserialize(node.value);
            if (newValue !== node.value) {
              node.update(newValue);
            }
          }
        };

        // Handle circular dependencies
        const placeholder = new RefPlaceholder<any>(contentRaw, factory);
        instances.set(contentRaw.__ref!, placeholder);
        await traverseNested();
        const final = await placeholder.final();
        instances.set(contentRaw.__ref!, final);
        return final;
      }
      return contentRaw;
    };

    const root = JSON.parse(raw);
    if (!isRootNode(root)) {
      throw new SerializerError("Provided data cannot be deserialized due to malformed format!");
    }

    await traverseObject(
      root,
      async ({ value: contentRaw, path }) => {
        output.update(path, await toDeserialize(contentRaw));
      },
      (_obj) => isSerializerNode(_obj),
    );
    return output.get();
  }

  static async deserialize<T = any>(
    raw: string,
    extraClasses?: SerializableClass<unknown>[],
  ): Promise<T> {
    const response = await Serializer.deserializeWithMeta(raw, extraClasses);
    return response.__root;
  }

  protected static _createOutputBuilder<T>() {
    return {
      _container: {} as T,
      get() {
        return this._container;
      },
      update(path: readonly string[], value: any) {
        setProp(this._container, path, value);
      },
      toJSON() {
        const seen = new SafeWeakSet();

        return JSON.stringify(this._container, (key, value) => {
          if (seen.has(value) && isSerializerNode(value)) {
            const updated: SerializerNode = {
              ...value,
              __value: SerializerRefIdentifier,
            };
            return updated;
          }

          seen.add(value);
          return value;
        });
      },
    };
  }
}

Serializer.register(Task<any>, {
  toPlain: (task) => ({ value: task.resolvedValue(), state: task.state }),
  fromPlain: ({ state, value }) => {
    const task = new Task();
    if (state === TaskState.RESOLVED) {
      task.resolve(value);
    } else if (state === TaskState.REJECTED) {
      task.reject(value);
    } else {
      task.reject(new SerializerError("Task cannot be solved due to serialization."));
    }
    task.resolve(value);
    return task;
  },
  createEmpty: () => new Task(),
  updateInstance: async (instance, value) => {
    instance.resolve(value);
  },
});
Serializer.register(SlidingTaskMap, {
  toPlain: (value) => ({
    config: {
      windowSize: value.windowSize,
      ttl: value.ttl,
    },
    entries: Array.from(value.entries()).filter(([_, task]) => {
      if (task instanceof Task) {
        return task.state === TaskState.RESOLVED;
      }
      return true;
    }),
  }),
  fromPlain: ({ entries, config }) => {
    const instance = new SlidingTaskMap(config.windowSize, config.ttl);
    for (const [key, value] of entries) {
      instance.set(key, value);
    }
    return instance;
  },
  createEmpty: () => new SlidingTaskMap(1, 1000),
  updateInstance: (instance, newInstance) => {
    Object.assign(instance, { windowSize: newInstance.windowSize, ttl: newInstance.ttl });
    newInstance.forEach((value, key) => instance.set(key, value));
  },
});
Serializer.register(Map, {
  toPlain: (value) => Array.from(value.entries()),
  fromPlain: (value) => new Map(value),
  createEmpty: () => new Map(),
  updateInstance: (instance, update) => {
    update.forEach(([key, value]) => instance.set(key, value));
  },
});
Serializer.register(Set, {
  toPlain: (value) => Array.from(value.values()),
  fromPlain: (value) => new Set(value),
  createEmpty: () => new Set(),
  updateInstance: (instance, update) => {
    update.forEach((value) => instance.add(value));
  },
});
Serializer.register(Array, {
  toPlain: (value) => value.slice(),
  fromPlain: (value) => value.slice(),
  createEmpty: () => [],
  updateInstance: (value, update) => {
    value.push(...update);
  },
});
Serializer.register(Object, {
  toPlain: (value) => Object.assign({}, value),
  fromPlain: (value) => Object.assign({}, value),
  createEmpty: () => ({}),
  updateInstance: (value, update) => Object.assign(value, update),
});
Serializer.register(Number, {
  toPlain: (value) => Number(value).toString(),
  fromPlain: (value) => Number(value),
});
Serializer.register(String, {
  toPlain: (value) => String(value),
  fromPlain: (value) => String(value),
});
Serializer.register(Boolean, {
  toPlain: (value) => Boolean(value),
  fromPlain: (value) => Boolean(value),
});
Serializer.register(BigInt, {
  toPlain: (value) => String(value),
  fromPlain: (value) => BigInt(value),
});
Serializer.register(Symbol, {
  toPlain: (value) => value.description,
  fromPlain: (value) => Symbol(value),
});
Serializer.register(Date, {
  toPlain: (value) => value.toISOString(),
  fromPlain: (value) => new Date(value),
});
Serializer.register(CacheFn, {
  toPlain: (value) => value.createSnapshot(),
  fromPlain: (value) => CacheFn.create(value.fn, value.options),
});
Serializer.register(Function, {
  toPlain: (value) => {
    const isNative = getProp(global, [value.name]) === value;

    function isConstructor(obj: any): obj is AnyConstructable {
      return obj && !!obj.prototype && !!obj.prototype.constructor.name;
    }

    return {
      name: value.name,
      binds: getFunctionBinds(value).map((bound) => ({
        name: isConstructor(bound) ? Serializer.findFactory(bound).targetClass : bound.name,
        value: "value" in bound ? bound.value : null,
      })),
      fn: isNative ? "" : String(value),
      isNative,
    };
  },
  fromPlain: (value) => {
    if (value.isNative) {
      return getProp(global, [value.name])!;
    }

    const toParsableForm = (): string => {
      let fn = value.fn;

      if (fn.match(/^\s*function.*?\(/)) {
        return fn;
      }

      const [a, b, c] = [fn.indexOf("=>"), fn.indexOf("("), fn.indexOf("{")];
      if (a > -1) {
        if (b === -1 || b > a || (c === -1 && c > a)) {
          const [p, p2] = halveString(fn, "=>", false);
          fn = `(${p.replace("async", "").replace("*", "").trim()})=>${p2}`;
          fn = [p.includes("async") && "async ", p.includes("*") && "*", fn]
            .filter(Boolean)
            .join(" ");
        }
      }

      const arrowStart = fn.indexOf("=>");
      const bracketStart = fn.indexOf("(");

      const [fnPrefix, fnContent = ""] =
        bracketStart === -1 || bracketStart > arrowStart
          ? halveString(fn, "{", true)
          : halveString(fn, "(", true);

      const nonReservedSymbols = fnPrefix
        .trim()
        .split(" ")
        .map((x) => x.trim())
        .filter(Boolean)
        .every((content) => ["async", "*"].includes(content));

      if (nonReservedSymbols) {
        return fn;
      }

      const name = value.name && fnPrefix.includes(value.name) ? value.name : "";
      let parameters = "";
      if (!fnContent.startsWith("(")) {
        if (fnPrefix.includes("(")) {
          parameters = (fnPrefix.match(/\((.+)\)/) ?? [null, ""])[1];
        } else {
          parameters = fnPrefix.replace("=>", "").replace("async", "").replace("*", "");
        }
      }

      return [
        fnPrefix.includes("async") && "async ",
        "function",
        fnPrefix.includes("*") && "*",
        name && ` ${name}`,
        parameters && `(${parameters})`,
        fnContent,
      ]
        .filter(Boolean)
        .join("");
    };

    const binds = value?.binds ?? [];
    const fn = Function(
      ...binds.map((b) => b.name),
      `return ${toParsableForm()}`,
    )(...binds.map((b) => (b.value ? b.value : Serializer.getFactory(b.name)!.ref)));

    if (value.name) {
      Object.defineProperty(
        fn,
        "name",
        Object.assign({}, Object.getOwnPropertyDescriptor(Object.getPrototypeOf(fn), "name"), {
          value: value.name,
        }),
      );
    }
    Object.defineProperty(
      fn,
      "toString",
      Object.assign({}, Object.getOwnPropertyDescriptor(Object.getPrototypeOf(fn), "name"), {
        value: () => value.fn,
      }),
    );
    return hasMinLength(binds, 1) ? toBoundedFunction(fn, binds) : fn;
  },
});

Serializer.register(Error, {
  toPlain: (value) => serializeError(value),
  fromPlain: (value) => deserializeError(value),
});
Serializer.register(RegExp, {
  toPlain: (value) => ({ source: value.source, flags: value.flags }),
  fromPlain: (value) => new RegExp(value.source, value.flags),
});
Serializer.register(WeakSet, {
  toPlain: () => {},
  fromPlain: () => new WeakSet(),
  createEmpty: () => new WeakSet(),
  updateInstance: () => {},
});
Serializer.register(WeakMap, {
  toPlain: () => {},
  fromPlain: () => new WeakMap(),
  createEmpty: () => new WeakMap(),
  updateInstance: () => {},
});
Serializer.register(WeakRef, {
  toPlain: () => {},
  fromPlain: () => new WeakRef({}),
});
Serializer.register(primitiveToSerializableClass(undefined), {
  toPlain: () => undefined,
  fromPlain: () => undefined,
});
Serializer.register(primitiveToSerializableClass(null), {
  toPlain: () => null,
  fromPlain: () => null,
});
// @ts-expect-error
Serializer.register(ZodType, {
  toPlain: (value) => toJsonSchema(value),
  fromPlain: () => {
    throw new Error("JSONSchema cannot be converted to zod!");
  },
});
Serializer.register(Buffer, {
  toPlain: (value: Buffer) => value.toString("base64"),
  fromPlain: (data) => Buffer.from(data, "base64"),
});
Serializer.register(AbortSignal, {
  toPlain: (value: AbortSignal) => ({
    aborted: value.aborted,
    reason: value.reason,
  }),
  fromPlain: (data) => {
    const controller = createAbortController();
    if (data.aborted) {
      controller.abort(data.reason);
    }
    return controller.signal;
  },
});
Serializer.register(AbortController, {
  toPlain: (value: AbortController) => ({
    signal: value.signal,
  }),
  fromPlain: (data) => createAbortController(data.signal),
});
