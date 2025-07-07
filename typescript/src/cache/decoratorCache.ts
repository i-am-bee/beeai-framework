/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyFn, TypedFn } from "@/internals/types.js";
import * as R from "remeda";
import hash from "object-hash";
import { createHash, createRandomHash } from "@/internals/helpers/hash.js";
import { findDescriptor } from "@/internals/helpers/prototype.js";

type InputFn = AnyFn;
type TargetFn = AnyFn;
type Instance = NonNullable<any>;

type CacheKeyFn = (this: Instance, ...args: any[]) => string;

export interface CacheDecoratorOptions {
  enabled: boolean;
  ttl?: number;
  cacheKey: CacheKeyFn;
  enumerable?: boolean;
}

interface CacheEntry {
  expiresAt: number;
  data: any;
}

interface FnContext {
  options: CacheDecoratorOptions;
  cache: Map<string, CacheEntry>;
}

interface GroupContext {
  options: CacheDecoratorOptions;
  instances: WeakMap<Instance, FnContext>;
}

export interface CacheDecoratorInstance {
  get(key?: string): CacheEntry | undefined;

  clear(keys?: string[]): void;

  isEnabled(): boolean;

  enable(): void;

  disable(): void;

  update(data: Partial<CacheDecoratorOptions>): void;
}

const state = {
  container: new WeakMap<InputFn, GroupContext>(),
  extractDescriptor(descriptor: PropertyDescriptor) {
    // Method approach
    if (descriptor.value != null) {
      return {
        value: descriptor.value as AnyFn,
        update: (value: TargetFn) => {
          descriptor.value = value;
        },
      };
    }
    // Accessor approach
    if (descriptor.get != null) {
      return {
        value: descriptor.get as AnyFn,
        update: (value: TargetFn) => {
          descriptor.get = value;
        },
      };
    }
    throw new Error(`@${Cache.name} decorator must be either on a method or get accessor.`);
  },
  getInstanceContext<T extends NonNullable<unknown>>(target: T, ctx: GroupContext) {
    if (!ctx.instances.has(target)) {
      ctx.instances.set(target, {
        cache: new Map(),
        options: ctx.options,
      });
    }
    return ctx.instances.get(target)!;
  },
} as const;

const initQueue = new WeakMap<Instance, { key: PropertyKey; enumerable: boolean }[]>();

export function Cache(_options?: Partial<CacheDecoratorOptions>) {
  const baseOptions: Required<CacheDecoratorOptions> = {
    enabled: true,
    cacheKey: ObjectHashKeyFn,
    ttl: Infinity,
    enumerable: false,
    ...R.pickBy(_options ?? {}, R.isDefined),
  };

  return function handler(
    obj: NonNullable<object>,
    key: string | symbol,
    descriptor: PropertyDescriptor,
  ) {
    if (Object.hasOwn(obj, "constructor")) {
      const constructor = obj.constructor;
      if (!initQueue.has(constructor)) {
        initQueue.set(constructor, []);
      }

      baseOptions.enumerable = Boolean(descriptor.get);
      initQueue
        .get(constructor)!
        .push({ key, enumerable: _options?.enumerable ?? baseOptions.enumerable });
    }

    const target = state.extractDescriptor(descriptor);
    if (descriptor.get && !_options?.cacheKey) {
      baseOptions.cacheKey = SingletonCacheKeyFn;
    }

    const groupContext: GroupContext = { instances: new WeakMap(), options: baseOptions };

    const fn = function wrapper(this: any, ...args: any[]) {
      const invokeOriginal = () => target.value.apply(this, args);
      const ctx = state.getInstanceContext(this, groupContext);
      if (!ctx.options.enabled) {
        return invokeOriginal();
      }

      const inputHash = ctx.options.cacheKey.apply(this, args);
      if (
        !ctx.cache.has(inputHash) ||
        (ctx.cache.get(inputHash)?.expiresAt ?? Infinity) < Date.now() // is expired check
      ) {
        const result = invokeOriginal();
        ctx.cache.set(inputHash, {
          expiresAt: Date.now() + (ctx.options.ttl ?? Infinity),
          data: result,
        });
      }
      return ctx.cache.get(inputHash)!.data;
    };
    Object.defineProperty(fn, "name", {
      get: () => target.value.name ?? "anonymous",
    });

    target.update(fn);
    state.container.set(fn, groupContext);
  };
}

Cache.init = function init<T extends NonNullable<unknown>>(self: T) {
  const task = initQueue.get(self.constructor) ?? [];
  for (const { key, enumerable } of task) {
    const descriptor = Object.getOwnPropertyDescriptor(self.constructor.prototype, key);
    if (descriptor) {
      Object.defineProperty(self, key, Object.assign(descriptor, { enumerable }));
    }
  }
  initQueue.delete(self);
};

Cache.getInstance = function getInstance<T extends NonNullable<unknown>>(
  target: T,
  property: keyof T,
): CacheDecoratorInstance {
  const descriptor = findDescriptor(target, property);
  if (!descriptor) {
    throw new TypeError(`No descriptor has been found for '${String(property)}'`);
  }
  const value = state.extractDescriptor(descriptor);
  const ctxByInstance = state.container.get(value.value);
  if (!ctxByInstance) {
    throw new TypeError(`No cache instance is bounded to '${String(property)}'!`);
  }

  const ctx = state.getInstanceContext(target, ctxByInstance);
  return {
    get(key = "") {
      return ctx.cache.get(key);
    },
    clear(keys?: string[]) {
      if (keys) {
        keys.forEach((key) => ctx.cache.delete(key));
      } else {
        ctx.cache.clear();
      }
    },
    update(data: CacheDecoratorOptions) {
      const oldTTL = ctx.options.ttl;
      const newTTL = data.ttl;
      if (oldTTL !== newTTL && newTTL !== undefined) {
        for (const value of ctx.cache.values()) {
          if (value.expiresAt === Infinity) {
            value.expiresAt = Date.now() + newTTL;
          } else {
            const diff = newTTL - (oldTTL ?? 0);
            value.expiresAt += diff;
          }
        }
      }
      Object.assign(ctx.options, data);
    },
    isEnabled() {
      return ctx.options.enabled;
    },
    enable() {
      ctx.options.enabled = true;
    },
    disable() {
      ctx.options.enabled = false;
    },
  };
};

export const WeakRefKeyFn = (() => {
  const _lookup = new WeakMap();

  const fn = (...args: any[]) => {
    const chunks = args.map((value) => {
      if (R.isObjectType(value) || R.isFunction(value)) {
        if (!_lookup.has(value)) {
          _lookup.set(value, createRandomHash(6));
        }
        return _lookup.get(value)!;
      }
      return value;
    });
    return createHash(JSON.stringify(chunks));
  };
  fn.from = <T>(cb: (self: T) => any[]) => {
    return function (this: T) {
      return cb(this).map(fn).join("#");
    };
  };
  return fn;
})() satisfies CacheKeyFn;

export const ObjectHashKeyFn: CacheKeyFn = (...args: any[]) =>
  hash(args, {
    encoding: "base64",
    replacer: (() => {
      const _lookup = new WeakMap();
      return (value: unknown) => {
        if (value && value instanceof AbortSignal) {
          // not supported by "hash" function
          if (!_lookup.has(value)) {
            _lookup.set(value, createRandomHash(6));
          }
          return _lookup.get(value)!;
        }
        return value;
      };
    })(),
    unorderedArrays: false,
    unorderedObjects: false,
    unorderedSets: false,
  });
export const JSONCacheKeyFn: CacheKeyFn = (...args: any[]) => JSON.stringify(args);
// eslint-disable-next-line unused-imports/no-unused-vars
export const SingletonCacheKeyFn: CacheKeyFn = (...args: any[]) => "";

export class CacheFn<P extends any[], R> extends Function {
  readonly name = CacheFn.name;

  static create<A extends any[], B>(
    fn: (...args: A) => B,
    options?: Partial<CacheDecoratorOptions>,
  ) {
    const instance = new CacheFn(fn, options);
    return instance as TypedFn<A, B> & CacheFn<A, B>;
  }

  constructor(
    protected readonly fn: (...args: P) => R,
    protected readonly options?: Partial<CacheDecoratorOptions>,
  ) {
    super();

    Cache.getInstance(this, "get").update(options ?? {});
    return new Proxy<typeof this>(this, {
      apply(this: CacheFn<P, R>, target, _, args: any[]) {
        return target.get(...(args as unknown as P));
      },
      get(target, prop: string | symbol, receiver: any) {
        const value = target[prop as keyof typeof target];
        if (value instanceof Function) {
          return function (this: CacheFn<P, R>, ...args: P) {
            return value.apply(this === receiver ? target : this, args);
          };
        }
        return value;
      },
    }) as unknown as CacheFn<P, R> & typeof fn;
  }

  updateTTL(ttl: number) {
    Cache.getInstance(this, "get").update({ ttl });
  }

  createSnapshot() {
    return {
      fn: this.fn,
      options: this.options,
    };
  }

  @Cache()
  get(...args: P) {
    return this.fn(...args);
  }
}
