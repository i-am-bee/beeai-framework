import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const agent = new ReActAgent({
  memory: new UnconstrainedMemory(),
  llm: new OllamaChatModel("llama3.1"),
  tools: [],
});
await agent.run({ prompt: "Hello world!" });

console.info(agent.memory.messages.length); // 2

const userMessage = agent.memory.messages[0];
console.info(`User: ${userMessage.text}`); // User: Hello world!

const agentMessage = agent.memory.messages[1];
console.info(`Agent: ${agentMessage.text}`); // Agent: Hello! It's nice to chat with you.
