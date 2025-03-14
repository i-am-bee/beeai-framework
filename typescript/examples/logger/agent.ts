import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Logger } from "beeai-framework/logger/logger";
import { Emitter } from "beeai-framework/emitter/emitter";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// Set up logging
Logger.defaults.pretty = true;

const logger = Logger.root.child({
  level: "trace",
  name: "app",
});

// Log events emitted during agent execution
Emitter.root.match("*.*", (data, event) => {
  const logLevel = event.path.includes(".run.") ? "trace" : "info";
  logger[logLevel](`Event '${event.path}' triggered by '${event.creator.constructor.name}'`);
});

// Create and run an agent
const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  memory: new UnconstrainedMemory(),
  tools: [],
});

const response = await agent.run({ prompt: "Hello!" });
logger.info(response.result.text);
