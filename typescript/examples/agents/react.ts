import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { createConsoleReader } from "../helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { Logger } from "beeai-framework/logger/logger";
import { PythonTool } from "beeai-framework/tools/python/python";
import { LocalPythonStorage } from "beeai-framework/tools/python/storage";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

Logger.root.level = "silent"; // disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

// Other models to try:
// "llama3.1:70b"
// "granite3.1-dense"
// "deepseek-r1:32b"
// ensure the model is pulled before running
const llm = new OllamaChatModel("llama3.1");

const codeInterpreterUrl = process.env.CODE_INTERPRETER_URL;
const __dirname = dirname(fileURLToPath(import.meta.url));

const codeInterpreterTmpdir =
  process.env.CODE_INTERPRETER_TMPDIR ?? "./examples/tmp/code_interpreter";
const localTmpdir = process.env.LOCAL_TMPDIR ?? "./examples/tmp/local";

const agent = new ReActAgent({
  llm,
  memory: new TokenMemory(),
  tools: [
    new DuckDuckGoSearchTool(),
    // new WebCrawlerTool(), // HTML web page crawler
    new WikipediaTool(),
    new OpenMeteoTool(), // weather tool
    // new ArXivTool(), // research papers
    // new DynamicTool() // custom python tool
    ...(codeInterpreterUrl
      ? [
          new PythonTool({
            codeInterpreter: { url: codeInterpreterUrl },
            storage: new LocalPythonStorage({
              interpreterWorkingDir: `${__dirname}/../../${codeInterpreterTmpdir}`,
              localWorkingDir: `${__dirname}/../../${localTmpdir}`,
            }),
          }),
        ]
      : []),
  ],
});

const reader = createConsoleReader();
if (codeInterpreterUrl) {
  reader.write(
    "🛠️ System",
    `The code interpreter tool is enabled. Please ensure that it is running on ${codeInterpreterUrl}`,
  );
}

try {
  for await (const { prompt } of reader) {
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
        // emitter.on("start", () => {
        //   reader.write(`Agent 🤖 : `, "starting new iteration");
        // });
        emitter.on("error", ({ error }) => {
          reader.write(`Agent 🤖 : `, FrameworkError.ensure(error).dump());
        });
        emitter.on("retry", () => {
          reader.write(`Agent 🤖 : `, "retrying the action...");
        });
        emitter.on("update", async ({ data, update, meta }) => {
          // log 'data' to see the whole state
          // to log only valid runs (no errors), check if meta.success === true
          reader.write(`Agent (${update.key}) 🤖 : `, update.value);
        });
        emitter.on("partialUpdate", ({ data, update, meta }) => {
          // ideal for streaming (line by line)
          // log 'data' to see the whole state
          // to log only valid runs (no errors), check if meta.success === true
          // reader.write(`Agent (partial ${update.key}) 🤖 : `, update.value);
        });

        // To observe all events (uncomment following block)
        // emitter.match("*.*", async (data: unknown, event) => {
        //   logger.trace(event, `Received event "${event.path}"`);
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
