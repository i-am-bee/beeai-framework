import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { GlobalTrajectoryMiddleware } from "beeai-framework/middleware/trajectory";
import { Tool } from "beeai-framework/tools/base";

const llm = new OllamaChatModel("granite4:micro");

// Create middleware to track all tool executions
const trajectoryMiddleware = new GlobalTrajectoryMiddleware({
  included: [Tool], // Only track tools
  pretty: false,
});

const agent = new ReActAgent({
  llm,
  memory: new TokenMemory(),
  tools: [new DuckDuckGoSearchTool(), new OpenMeteoTool()],
});

const response = await agent
  .run({ prompt: "What's the current weather in Las Vegas?" })
  .middleware(trajectoryMiddleware);

console.log(`\nAgent 🤖 : `, response.result.text);
