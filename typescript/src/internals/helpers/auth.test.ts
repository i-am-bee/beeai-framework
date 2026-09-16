/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { isApiKeyValid } from "@/internals/helpers/auth.js";

describe("isApiKeyValid", () => {
  it("treats a missing configured key as auth disabled", () => {
    expect(isApiKeyValid(undefined, undefined)).toBe(true);
    expect(isApiKeyValid(null, "anything")).toBe(true);
    expect(isApiKeyValid("", "anything")).toBe(true);
  });

  it("rejects a missing supplied key", () => {
    expect(isApiKeyValid("secret", undefined)).toBe(false);
    expect(isApiKeyValid("secret", "")).toBe(false);
  });

  it("accepts a matching key and rejects a mismatch", () => {
    expect(isApiKeyValid("secret", "secret")).toBe(true);
    expect(isApiKeyValid("secret", "wrong")).toBe(false);
  });

  it("strips the Bearer prefix only when asked", () => {
    expect(isApiKeyValid("secret", "Bearer secret", { stripBearerPrefix: true })).toBe(true);
    expect(isApiKeyValid("secret", "Bearer secret")).toBe(false);
  });

  it.each(["bearer secret", "BEARER secret", "Bearer    secret"])(
    "handles the Bearer scheme case-insensitively: %s",
    (header) => {
      expect(isApiKeyValid("secret", header, { stripBearerPrefix: true })).toBe(true);
    },
  );

  it("strips only the leading prefix, not a match inside the key", () => {
    const key = "sk-Bearer team1";
    expect(isApiKeyValid(key, `Bearer ${key}`, { stripBearerPrefix: true })).toBe(true);
  });

  it("does not throw when the lengths differ", () => {
    expect(() => isApiKeyValid("short", "a-much-longer-token")).not.toThrow();
    expect(isApiKeyValid("short", "a-much-longer-token")).toBe(false);
  });
});
