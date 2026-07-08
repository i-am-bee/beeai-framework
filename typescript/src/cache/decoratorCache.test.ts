/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Cache,
  SingletonCacheKeyFn,
  JSONCacheKeyFn,
  ObjectHashKeyFn,
  CacheFn,
} from "@/cache/decoratorCache.js";
import type { AnyFn } from "@/internals/types.js";

describe("@Cache decorator", () => {
  it("Caches method", () => {
    class A {
      static number = 0;

      @Cache()
      getNumber() {
        A.number += 1;
        return A.number;
      }
    }

    const a = new A();
    expect(a.getNumber()).toEqual(1);
    expect(a.getNumber()).toEqual(1);
  });

  it("Caches a complex datatype", () => {
    class A {
      static number = 0;

      @Cache({
        cacheKey: ObjectHashKeyFn,
      })
      getNumber(handler: AnyFn) {
        A.number += 1;
        return handler;
      }
    }

    const a = new A();
    const x = () => 1;
    const y = () => 1;

    expect(a.getNumber(x)).toEqual(x);
    expect(A.number).toBe(1);
    expect(a.getNumber(y)).toEqual(y);
    expect(A.number).toBe(2);
    expect(a.getNumber(x)).toEqual(x);
    expect(A.number).toBe(2);
    expect(a.getNumber(y)).toEqual(y);
    expect(A.number).toBe(2);
  });

  it("Ignores args if 'EmptyCacheKeyFn' is specified.", () => {
    class A {
      static number = 0;

      @Cache({
        cacheKey: SingletonCacheKeyFn,
      })
      // eslint-disable-next-line unused-imports/no-unused-vars
      getNumber(...args: any[]) {
        A.number += 1;
        return A.number;
      }
    }

    const a = new A();
    expect(a.getNumber("a", "b")).toEqual(1);
    expect(a.getNumber("b", "c")).toEqual(1);
  });

  it("Caches an accessor", () => {
    class A {
      static number = 0;

      @Cache()
      get getNumber() {
        A.number += 1;
        return A.number;
      }
    }

    const a = new A();
    expect(a.getNumber).toEqual(1);
    expect(a.getNumber).toEqual(1);
  });

  it("Controls cache behaviour", () => {
    class A {
      constructor(public invocations = 0) {}

      @Cache()
      get getNumber() {
        this.invocations += 1;
        return this.invocations;
      }
    }

    const a = new A();
    const cache = Cache.getInstance(a, "getNumber");
    cache.disable();
    expect(a.getNumber).toEqual(1);
    expect(a.getNumber).toEqual(2);
    expect(a.getNumber).toEqual(3);
    cache.enable();
    expect(a.getNumber).toEqual(4);
    expect(a.getNumber).toEqual(4);
  });

  it("Clears cache", () => {
    class A {
      static number = 0;

      @Cache()
      get getNumber() {
        A.number += 1;
        return A.number;
      }

      reset() {
        Cache.getInstance(this, "getNumber").clear();
      }
    }

    const a = new A();
    expect(a.getNumber).toEqual(1);
    expect(a.getNumber).toEqual(1);
    a.reset();
    expect(a.getNumber).toEqual(2);
  });

  it("Clears cache for inherited members", () => {
    class A {
      constructor(public readonly results: number[]) {}

      @Cache()
      toString() {
        return this.results.join(",");
      }
    }

    class B extends A {}

    const instance = new B([1, 2, 3]);
    expect(instance.toString()).toEqual("1,2,3");
    instance.results.length = 0;
    expect(instance.toString()).toEqual("1,2,3");
    instance.results.push(4);
    Cache.getInstance(instance, "toString").clear();
    expect(instance.toString()).toEqual("4");
  });

  it("Clears cache by a key", () => {
    class A {
      static calls = 0;

      @Cache({
        cacheKey: JSONCacheKeyFn,
      })
      getNumber(a: number) {
        A.calls += a;
        return A.calls;
      }

      reset(values: number[]) {
        Cache.getInstance(this, "getNumber").clear(values.map((value) => JSONCacheKeyFn(value)));
      }
    }

    const a = new A();
    expect(a.getNumber(10)).toEqual(10);
    expect(a.getNumber(10)).toEqual(10);
    expect(a.getNumber(20)).toEqual(30);
    expect(a.getNumber(20)).toEqual(30);
    a.reset([10]);
    expect(a.getNumber(10)).toEqual(40);
    expect(a.getNumber(20)).toEqual(30);
  });

  it("Caches a complex input", () => {
    class A {
      static calls = 0;

      @Cache()
      calculate(a: number, b: number) {
        A.calls += 1;
        return a + b;
      }
    }

    const a = new A();
    expect(a.calculate(2, 3)).toEqual(5);
    expect(a.calculate(2, 3)).toEqual(5);
    expect(A.calls).toEqual(1);
  });

  it("Does not interfere cache across instances", () => {
    class A {
      constructor(public readonly base: number) {}

      @Cache()
      add(input: number) {
        return input + this.base;
      }
    }

    const a = new A(100);
    const b = new A(200);

    expect(a.add(1)).toEqual(101);
    expect(a.add(2)).toEqual(102);

    expect(b.add(1)).toEqual(201);
    expect(b.add(2)).toEqual(202);
  });

  it("Preserves meta information", () => {
    class A {
      @Cache()
      calculate() {}
    }

    const a = new A();
    const b = new A();
    expect(a.calculate.name).toBe("calculate");
    expect(b.calculate.name).toBe("calculate");
  });

  it("Static caching", () => {
    class A {
      static counter = 0;

      @Cache()
      static add(input: number) {
        A.counter += 1;
        return input;
      }

      static bbb(input: number) {
        A.counter += 1;
        return input;
      }
    }

    class B extends A {}

    expect(A.add(1)).toEqual(1);
    expect(A.add(1)).toEqual(1);
    expect(B.add(1)).toEqual(1);
    expect(B.add(1)).toEqual(1);

    expect(A.counter).toBe(2);
    expect(B.counter).toBe(2);
  });

  it("Retrieves cache entry directly", async () => {
    class A {
      @Cache()
      get instance() {
        return Promise.resolve(1);
      }
    }
    const a = new A();
    const cache = Cache.getInstance(a, "instance");
    expect(cache.get()).toBeUndefined();
    await a.instance;
    expect(cache.get()).toMatchObject({
      expiresAt: Infinity,
      data: expect.any(Promise),
    });
  });

  describe("CacheFn", () => {
    it("Caches", () => {
      const fn = vi.fn((input) => input);

      const cached = CacheFn.create(fn);
      expect(cached(1)).toBe(1);
      expect(cached(1)).toBe(1);
      expect(cached(2)).toBe(2);
      expect(cached(2)).toBe(2);
      expect(cached(3)).toBe(3);
      expect(cached(3)).toBe(3);

      expect(fn).toHaveBeenCalledTimes(3);
    });

    it("Updates TTL", async () => {
      const sleep = async (ms: number) => await new Promise((resolve) => setTimeout(resolve, ms));
      let counter = 0;
      const cached = CacheFn.create(
        vi.fn(async () => {
          await sleep(50);
          cached.updateTTL(200);
          counter += 1;
          return counter;
        }),
      );
      await expect(cached()).resolves.toBe(1);
      await expect(cached()).resolves.toBe(1);
      await sleep(250);
      await expect(cached()).resolves.toBe(2);
      await expect(cached()).resolves.toBe(2);
    });
  });
});
