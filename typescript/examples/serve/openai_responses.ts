import "dotenv/config.js";

import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OpenAIServer } from "beeai-framework/adapters/openai/serve/server";

// ensure the model is pulled before running
const llm = new OllamaChatModel("granite4:micro");

const agent = new ToolCallingAgent({
  llm,
  memory: new UnconstrainedMemory(),
  tools: [
    new OpenMeteoTool(), // weather tool
  ],
});

await new OpenAIServer({ api: "responses", host: "0.0.0.0", port: 9999 }).register(agent).serve();
