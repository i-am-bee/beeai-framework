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

import { parseEnv } from "@/internals/env.js";
import { z } from "zod";

describe("Parsing ENV", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("Correctly parses a string", () => {
    vi.stubEnv("LOG_LEVEL", "info");
    expect(parseEnv("LOG_LEVEL", z.string().min(1))).toBe("info");
  });

  it("Correctly parses a boolean", () => {
    vi.stubEnv("ENABLE_LOGGING", "  true");
    expect(parseEnv.asBoolean("ENABLE_LOGGING")).toBe(true);

    vi.stubEnv("ENABLE_LOGGING", "false");
    expect(parseEnv.asBoolean("ENABLE_LOGGING")).toBe(false);
  });
});
