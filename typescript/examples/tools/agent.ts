/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ArXivTool } from "beeai-framework/tools/arxiv";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  memory: new UnconstrainedMemory(),
  tools: [new ArXivTool()],
});
