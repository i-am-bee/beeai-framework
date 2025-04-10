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
