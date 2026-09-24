import "dotenv/config.js";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { HandoffTool } from "beeai-framework/tools/handoff";
import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ConditionalRequirement } from "beeai-framework/agents/requirement/requirements/conditional";
import { GlobalTrajectoryMiddleware } from "beeai-framework/middleware/trajectory";
import { Tool } from "beeai-framework/tools/base";

const llm = new OllamaChatModel("granite4:micro");

const agentB = new RequirementAgent({
  llm,
  memory: new TokenMemory(),
  tools: [new OpenMeteoTool()],
  requirements: [new ConditionalRequirement(OpenMeteoTool, { forceAtStep: 1 })],
});

const agentA = new RequirementAgent({
  llm,
  memory: new TokenMemory(),
  tools: [new HandoffTool(agentB)],
  requirements: [new ConditionalRequirement(HandoffTool, { forceAtStep: 1 })],
});

const response = await agentA
  .run({ prompt: "What's the current weather in Las Vegas?" })
  .middleware(
    new GlobalTrajectoryMiddleware({
      included: [Tool],
    }),
  );

console.log(`Agent 🤖 : `, response.result.text);
