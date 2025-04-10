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
