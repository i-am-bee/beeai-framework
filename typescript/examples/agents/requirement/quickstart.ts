import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ConditionalRequirement } from "beeai-framework/agents/requirement/requirements/conditional";
import { ChatModel } from "beeai-framework/backend/chat";
import { ThinkTool } from "beeai-framework/tools/think";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { Tool } from "beeai-framework/tools/base";
import { GlobalTrajectoryMiddleware } from "beeai-framework/middleware/trajectory";

// Create an agent that plans activities based on weather and events
const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:llama3.1"),
  tools: [
    new ThinkTool(), // to reason
    new OpenMeteoTool(), // retrieve weather data
    new WikipediaTool(), // search for data
  ],
  instructions: "Plan activities for a given destination based on current weather.",
  requirements: [
    // Force thinking first
    new ConditionalRequirement(ThinkTool, { forceAtStep: 1, maxInvocations: 5 }),
    // Search only after getting weather and at least once
    new ConditionalRequirement(WikipediaTool, {
      onlyAfter: [OpenMeteoTool],
      minInvocations: 1,
      maxInvocations: 2,
    }),
    // Weather tool be used at least once but not consecutively
    new ConditionalRequirement(OpenMeteoTool, {
      consecutiveAllowed: false,
      minInvocations: 1,
      maxInvocations: 2,
    }),
  ],
  middlewares: [
    new GlobalTrajectoryMiddleware({
      included: [Tool],
    }),
  ],
});

// Run with execution logging
const response = await agent.run({ prompt: "What to do in Boston?" });

console.log(`Final Answer: ${response.result.text}`);
