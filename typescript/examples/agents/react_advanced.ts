import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { createConsoleReader } from "../helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { Logger } from "beeai-framework/logger/logger";
import {
  DuckDuckGoSearchTool,
  DuckDuckGoSearchToolSearchType,
} from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { z } from "zod";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

Logger.root.level = "silent"; // disable internal logs
const logger = new Logger({ name: "app", level: "trace" });

const llm = new OllamaChatModel("llama3.1");

const agent = new ReActAgent({
  llm,
  memory: new UnconstrainedMemory(),
  // You can override internal templates
  templates: {
    user: (template) =>
      template.fork((config) => {
        config.schema = z.object({ input: z.string() }).passthrough();
        config.template = `User: {{input}}`;
      }),
    system: (template) =>
      template.fork((config) => {
        config.defaults.instructions =
          "You are a helpful assistant that uses tools to answer questions.";
      }),
    toolNoResultError: (template) =>
      template.fork((config) => {
        config.template += `\nPlease reformat your input.`;
      }),
    toolNotFoundError: (template) =>
      template.fork((config) => {
        config.schema = z
          .object({ tools: z.array(z.object({ name: z.string() }).passthrough()) })
          .passthrough();
        config.template = `Tool does not exist!
{{#tools.length}}
Use one of the following tools: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.length}}`;
      }),
  },
  tools: [
    new DuckDuckGoSearchTool({
      maxResults: 10,
      search: {
        safeSearch: DuckDuckGoSearchToolSearchType.STRICT,
      },
    }),
    // new WebCrawlerTool(), // HTML web page crawler
    // new WikipediaTool(),
    new OpenMeteoTool(), // weather tool
    // new ArXivTool(), // research papers
    // new DynamicTool() // custom python tool
  ],
});

const reader = createConsoleReader();

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
          signal: AbortSignal.timeout(2 * 60 * 1000),
        },
      )
      .observe((emitter) => {
        emitter.on("start", () => {
          reader.write(`Agent 🤖 : `, "starting new iteration");
        });
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
      });

    reader.write(`Agent 🤖 : `, response.result.text);
  }
} catch (error) {
  logger.error(FrameworkError.ensure(error).dump());
} finally {
  reader.close();
}
