import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { GlobalTrajectoryMiddleware } from "beeai-framework/middleware";
import { BaseAgent } from "beeai-framework/agents/base";
import { ChatModel } from "beeai-framework/backend/chat";
import { Tool } from "beeai-framework/tools/base";

const llm = new OllamaChatModel("llama3.1");

// Create middleware with custom formatting
const trajectoryMiddleware = new GlobalTrajectoryMiddleware({
  target: "console",
  pretty: true,
  formatter: (input) => {
    return `[${input.className}] ${input.instanceName || "unnamed"} - ${input.eventName}`;
  },
  prefixByType: new Map([
    [BaseAgent, "🐝 "],
    [ChatModel, "🧠 "],
    [Tool, "⚙️ "],
  ]),
});

const agent = new ReActAgent({
  llm,
  memory: new TokenMemory(),
  tools: [new DuckDuckGoSearchTool()],
});

const response = await agent
  .run({ prompt: "What is the capital of France?" })
  .middleware(trajectoryMiddleware);

console.log(`\nAgent 🤖 : `, response.result.text);
