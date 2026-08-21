import "dotenv/config";
import { FeedoSearchTool } from "beeai-framework/tools/search/feedo";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

async function main() {
  if (!process.env.FEEDO_USAGE_KEY) {
    console.error("Please set the FEEDO_USAGE_KEY environment variable.");
    console.error("Example: export FEEDO_USAGE_KEY='0x123...'");
    console.error("You can get a free testnet key at https://feedo.ink");
    process.exit(1);
  }

  // 1. Initialize the Feedo tool
  console.log("Initializing Feedo Memory Tool...");
  const feedoTool = new FeedoSearchTool();

  // 2. You can run the tool directly
  console.log("Testing direct tool call...");
  const result = await feedoTool.run({
    action: "add",
    query: "The project codename is 'Apollo'.",
    topic: "project"
  });
  console.log(`Direct result: ${result.getTextContent()}`);

  // 3. Or pass it to an Agent so it can autonomously manage memory
  console.log("\nInitializing ReActAgent with Feedo Tool...");
  const agent = new ReActAgent({
    llm: new OllamaChatModel("granite4:micro"),
    memory: new UnconstrainedMemory(),
    tools: [feedoTool],
  });

  console.log("\nAsking agent to recall the project name...");
  const response = await agent.run({
    prompt: "What is the project codename? If you don't know, search your memory.",
  });
  console.log(`Agent Response: ${response.result.text}`);
}

main().catch(console.error);
