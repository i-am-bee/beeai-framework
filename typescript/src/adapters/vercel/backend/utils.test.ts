/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { parseHeadersFromEnv } from "./utils.js";

describe("parseHeadersFromEnv", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it.each([
    ["Authorization=Bearer abc==", { Authorization: "Bearer abc==" }],
    ["X-Token=key=value=tail", { "X-Token": "key=value=tail" }],
    [" X-Token = abc== , X-Other = value ", { "X-Token": "abc==", "X-Other": "value" }],
    ["X-Empty=,invalid,X-Other=value", { "X-Empty": "", "X-Other": "value" }],
    ["", {}],
    [undefined, {}],
  ])("parses %s", (value, expected) => {
    vi.stubEnv("OPENAI_API_HEADERS", value);
    expect(parseHeadersFromEnv("OPENAI_API_HEADERS")).toEqual(expected);
  });
});
