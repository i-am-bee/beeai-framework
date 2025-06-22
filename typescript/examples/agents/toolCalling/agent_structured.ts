import "dotenv/config.js";
import { createConsoleReader } from "../../helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { Logger } from "beeai-framework/logger/logger";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { z } from "zod";

Logger.root.level = "silent"; // disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

// Other models to try:
// "llama3.1:70b"
// "granite3.3"
// "deepseek-r1:32b"
// ensure the model is pulled before running
const llm = new OllamaChatModel("llama3.1:8b");

const agent = new ToolCallingAgent({
  llm,
  memory: new TokenMemory(),
  tools: [
    new OpenMeteoTool(), // weather tool
  ],
});

const reader = createConsoleReader();

try {
  const schema = z.object({
    firstName: z.string().min(1),
    lastName: z.string().min(1),
    age: z.number().min(1).max(99),
    country: z.string(),
  });

  const response = await agent.run({
    prompt: "Generate profile of a citizen.",
    expectedOutput: schema,
  });
  console.info(response.result.text); // JSON text

  const raw = JSON.parse(response.result.text);
  const parsed = schema.parse(raw);
  console.info(parsed); // parsed and validated schema instance
} catch (error) {
  logger.error(FrameworkError.ensure(error).dump());
} finally {
  reader.close();
}
