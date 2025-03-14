import "dotenv/config.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { OpenAPITool } from "beeai-framework/tools/openapi";
import * as fs from "fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { ChatModel } from "beeai-framework/backend/chat";

const llm = await ChatModel.fromName("ollama:llama3.1");

const __dirname = dirname(fileURLToPath(import.meta.url));
const openApiSchema = await fs.promises.readFile(
  `${__dirname}/assets/github_openapi.json`,
  "utf-8",
);

const agent = new ReActAgent({
  llm,
  memory: new TokenMemory(),
  tools: [new OpenAPITool({ openApiSchema })],
});

const response = await agent
  .run({ prompt: 'How many repositories are in "i-am-bee" org?' })
  .observe((emitter) => {
    emitter.on("update", async ({ data, update, meta }) => {
      console.log(`Agent (${update.key}) 🤖 : `, update.value);
    });
  });

console.log(`Agent 🤖 : `, response.result.text);
