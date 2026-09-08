/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { setProp } from "@/internals/helpers/object.js";

describe("setProp", () => {
  it("sets a shallow value", () => {
    const target: any = {};
    setProp(target, ["a"], 1);
    expect(target).toEqual({ a: 1 });
  });

  it("creates intermediate objects for a nested path", () => {
    const target: any = {};
    setProp(target, ["a", "b", "c"], "x");
    expect(target).toEqual({ a: { b: { c: "x" } } });
  });

  it("writes into arrays without disturbing length", () => {
    const target: any = [1, 2, 3];
    setProp(target, [1], "y");
    expect(target).toEqual([1, "y", 3]);
    expect(target.length).toBe(3);
  });

  it("rejects __proto__ as a path key", () => {
    const target: any = {};
    expect(() => setProp(target, ["__proto__", "polluted"], true)).toThrowError(TypeError);
    expect(({} as any).polluted).toBeUndefined();
  });

  describe("prototype integrity", () => {
    afterEach(() => {
      delete (Object.prototype as any).intercepted;
      delete (Object.prototype as any).polluted;
    });

    it("does not reach Object.prototype via constructor/prototype", () => {
      const target: any = {};
      setProp(target, ["constructor", "prototype", "polluted"], true);

      // The walk creates plain own properties instead of following the inherited
      // constructor, so nothing escapes onto Object.prototype.
      expect(({} as any).polluted).toBeUndefined();
      expect(Object.hasOwn(target, "constructor")).toBe(true);
    });

    it("ignores an inherited setter and always creates an own property", () => {
      let sideEffect: unknown = undefined;
      Object.defineProperty(Object.prototype, "intercepted", {
        configurable: true,
        set(v: unknown) {
          sideEffect = v;
        },
      });

      const target: any = {};
      setProp(target, ["intercepted"], "value");

      // Object.assign would have invoked the inherited setter and created no own
      // property; defineProperty writes straight onto the target.
      expect(sideEffect).toBeUndefined();
      expect(Object.hasOwn(target, "intercepted")).toBe(true);
      expect(target.intercepted).toBe("value");
    });
  });

  describe("legitimate data keys", () => {
    it("allows 'constructor' and 'prototype' as ordinary leaf keys", () => {
      // These are valid property names in user data -- e.g. deepCopy() feeds setProp
      // arbitrary keys from traverseObject() -- so they must not be blanket-rejected.
      const target: any = {};
      setProp(target, ["constructor"], "legit");
      setProp(target, ["prototype"], 42);

      expect(target.constructor).toBe("legit");
      expect(target.prototype).toBe(42);
      expect(({} as any).constructor).toBe(Object);
    });
  });
});
