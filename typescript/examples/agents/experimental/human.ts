import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { createConsoleReader } from "../../helpers/io.js"; // Use the examples console reader
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { Logger } from "beeai-framework/logger/logger";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

// Import the HumanTool from the updated file
import { HumanTool } from "../../tools/experimental/human.js";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// Set up logger
Logger.root.level = "silent"; // Disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

// Initialize LLM (test against llama as requested)
const llm = new OllamaChatModel("llama3.1");

// Create the console reader once, share it with HumanTool
const reader = createConsoleReader();

// Initialize ReActAgent with shared reader for HumanTool
const agent = new ReActAgent({
  llm,
  memory: new TokenMemory(),
  tools: [
    new OpenMeteoTool(),
    new HumanTool({
      reader: reader,
    }),
  ],
});

// Main loop
try {
  for await (const { prompt } of reader) {
    // Run the agent and observe events
    const response = await agent
      .run(
        { prompt },
        {
          execution: {
            maxRetriesPerStep: 3,
            totalMaxRetries: 10,
            maxIterations: 20,
          },
        },
      )
      .observe((emitter) => {
        // Show only final answers
        emitter.on("update", async ({ update }) => {
          if (update.key === "final_answer") {
            reader.write("Agent 🤖 : ", update.value);
          }
        });

        // Log errors
        emitter.on("error", ({ error }) => {
          reader.write("Agent 🤖 : ", FrameworkError.ensure(error).dump());
        });

        // Retry notifications
        emitter.on("retry", () => {
          reader.write("Agent 🤖 : ", "Retrying the action...");
        });
      });

    // Print the final response
    if (response.result?.text) {
      reader.write("Agent 🤖 : ", response.result.text);
    } else {
      reader.write(
        "Agent 🤖 : ",
        "No result was returned. Ensure your input is valid or check tool configurations.",
      );
    }
  }
} catch (error) {
  logger.error(FrameworkError.ensure(error).dump());
} finally {
  // Gracefully close the reader when exiting the app
  reader.close();
}
