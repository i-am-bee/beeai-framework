/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { SlidingCache } from "@/cache/slidingCache.js";
import { beforeEach } from "vitest";

describe("SlidingCache", () => {
  beforeEach(() => {
    vitest.useFakeTimers();
  });

  it("Handles basic operations", async () => {
    const instance = new SlidingCache({
      ttl: 5000,
      size: 10,
    });
    await instance.set("1", 1);
    await instance.set("2", 2);
    await expect(instance.has("1")).resolves.toBe(true);
    await expect(instance.get("1")).resolves.toBe(1);
    await expect(instance.size()).resolves.toBe(2);
    await instance.delete("1");
    await expect(instance.size()).resolves.toBe(1);
    await instance.clear();
    await expect(instance.size()).resolves.toBe(0);
  });

  it("Removes old entries", async () => {
    const instance = new SlidingCache({
      ttl: 2500,
      size: 10,
    });

    await instance.set("1", 1);
    await vitest.advanceTimersByTimeAsync(1500);
    await expect(instance.size()).resolves.toBe(1);
    await expect(instance.has("1")).resolves.toBe(true);
    await expect(instance.get("1")).resolves.toBe(1);

    await vitest.advanceTimersByTimeAsync(2000);
    await expect(instance.has("1")).resolves.toBe(false);
    await expect(instance.get("1")).resolves.toBe(undefined);

    await instance.clear();
  });

  it("Serializes", async () => {
    const instance = new SlidingCache({
      ttl: 5000,
      size: 10,
    });
    await instance.set("1", 1);
    await instance.set("2", 2);
    await instance.set("3", 3);
    const serialized = await instance.serialize();
    const deserialized = await SlidingCache.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
