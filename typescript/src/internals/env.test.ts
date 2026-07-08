/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
