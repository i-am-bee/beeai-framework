import "dotenv/config.js";
import { createConsoleReader } from "../../helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { Logger } from "beeai-framework/logger/logger";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";

Logger.root.level = "silent"; // disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

// Other models to try:
// "llama3.1:70b"
// "granite3.3"
// "deepseek-r1:32b"
// ensure the model is pulled before running
const llm = new OllamaChatModel("llama3.1");

const agent = new ToolCallingAgent({
  llm,
  memory: new TokenMemory(),
  tools: [
    new OpenMeteoTool(), // weather tool
  ],
});

const reader = createConsoleReader();

try {
  for await (const { prompt } of reader) {
    let messagesCount = agent.memory.messages.length + 1;

    const response = await agent.run({ prompt }).observe((emitter) => {
      emitter.on("success", async ({ state }) => {
        const newMessages = state.memory.messages.slice(messagesCount);
        messagesCount += newMessages.length;

        reader.write(
          `Agent (${newMessages.length} new messages) 🤖 :\n`,
          newMessages.map((msg) => `-> ${JSON.stringify(msg.toPlain())}`).join("\n"),
        );
      });

      // To observe all events (uncomment following block)
      // emitter.match("*.*", async (data: unknown, event) => {
      //   logger.trace(event, `Received event "${event.path}"`);
      // }, {
      //   matchNested: true
      // });

      // To get raw LLM input (uncomment following block)
      // emitter.match(
      //   (event) => event.creator === llm && event.name === "start",
      //   async (data: InferCallbackValue<GenerateEvents["start"]>, event) => {
      //     logger.trace(
      //       event,
      //       [
      //         `Received LLM event "${event.path}"`,
      //         JSON.stringify(data.input), // array of messages
      //       ].join("\n"),
      //     );
      //   },
      // );
    });

    reader.write(`Agent 🤖 : `, response.result.text);
  }
} catch (error) {
  logger.error(FrameworkError.ensure(error).dump());
} finally {
  reader.close();
}
