/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { beforeEach } from "vitest";
import { UnconstrainedCache } from "@/cache/unconstrainedCache.js";

describe("UnconstrainedCache", () => {
  beforeEach(() => {
    vitest.useFakeTimers();
  });

  it("Handles basic operations", async () => {
    const instance = new UnconstrainedCache();
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

  it("Serializes", async () => {
    const instance = new UnconstrainedCache();
    await instance.set("1", 1);
    const serialized = await instance.serialize();
    const deserialized = await UnconstrainedCache.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
