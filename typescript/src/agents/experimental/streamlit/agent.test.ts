/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OllamaChatModel } from "@/adapters/ollama/backend/chat.js";
import { StreamlitAgent } from "./agent.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";

describe("Streamlit agent", () => {
  it("Serializes", async () => {
    const instance = new StreamlitAgent({
      llm: new OllamaChatModel("llama3.1"),
      memory: new UnconstrainedMemory(),
    });
    const serialized = await instance.serialize();
    const deserialized = await StreamlitAgent.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
