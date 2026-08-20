/**
 * Copyright 2025 BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { LLMTool } from "./llm.js";

describe("LLMTool", () => {
  it("renders the default task prompt as a coherent sentence", () => {
    const rendered = LLMTool.template.render({ task: "Summarize the notes" });

    expect(rendered).toContain(
      "Use common sense and the information contained in the conversation up to this point to complete the following task.",
    );
    expect(rendered).not.toContain("using Using");
    expect(rendered).toContain("The Task: Summarize the notes");
  });
});
