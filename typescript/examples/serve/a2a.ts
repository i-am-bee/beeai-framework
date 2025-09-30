import "dotenv/config.js";

import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { A2AServer } from "beeai-framework/adapters/a2a/serve/server";

// ensure the model is pulled before running
const llm = new OllamaChatModel("llama3.1");

const agent = new ToolCallingAgent({
  llm,
  memory: new UnconstrainedMemory(),
  tools: [
    new OpenMeteoTool(), // weather tool
  ],
});

await new A2AServer().register(agent).serve();
