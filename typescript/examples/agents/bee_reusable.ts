import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// We create an agent
let agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  tools: [new WikipediaTool()],
  memory: new UnconstrainedMemory(),
});

// We ask the agent
let prompt = "Who is the president of USA?";
console.info(prompt);
const response = await agent.run({
  prompt,
});
console.info(response.result.text);

// We can save (serialize) the agent
const json = await agent.serialize();

// We reinitialize the agent to the exact state he was
agent = await ReActAgent.fromSerialized(json);

// We continue in our conversation
prompt = "When was he born?";
console.info(prompt);
const response2 = await agent.run({
  prompt,
});
console.info(response2.result.text);
