/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { NullCache } from "@/cache/nullCache.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";

describe("NullCache", () => {
  it("Handles basic operations", async () => {
    const instance = new NullCache();
    await instance.set("1", 1);
    await expect(instance.has("1")).resolves.toBe(false);
    await expect(instance.get("1")).resolves.toBe(undefined);
    await instance.delete("1");
    await expect(instance.size()).resolves.toBe(0);
    await instance.clear();
  });

  it("Serializes", async () => {
    const instance = new NullCache();
    await instance.set("1", 1);
    const serialized = await instance.serialize();
    const deserialized = await NullCache.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
