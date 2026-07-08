import "dotenv/config.js";
import { FrameworkError } from "beeai-framework/errors";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { createConsoleReader } from "examples/helpers/io.js";
import { RePlanAgent } from "beeai-framework/agents/experimental/replan/agent";
import { WatsonxChatModel } from "beeai-framework/adapters/watsonx/backend/chat";

const reader = createConsoleReader();

const agent = new RePlanAgent({
  llm: new WatsonxChatModel("meta-llama/llama-3-3-70b-instruct"),
  memory: new UnconstrainedMemory(),
  tools: [new DuckDuckGoSearchTool(), new OpenMeteoTool()],
});

try {
  for await (const { prompt } of reader) {
    const response = await agent.run({ prompt }).observe((emitter) => {
      emitter.on("update", async ({ state }) => {
        reader.write("💭 ", state.lookback);
        state.plan.forEach((step) => reader.write("➡️ ", step.title));
      });
      emitter.on("tool", (data) => {
        if (data.type === "start") {
          reader.write(`🛠️ `, `Start ${data.tool.name} with ${JSON.stringify(data.input)}`);
        } else if (data.type === "success") {
          reader.write(`🛠 `, `Success ${data.tool.name} with ${JSON.stringify(data.output)}`);
        } else if (data.type === "error") {
          reader.write(
            `🛠 Error ${data.tool.name}`,
            `with ${FrameworkError.ensure(data.error).dump()}`,
          );
        }
      });
    });
    reader.write(`Agent 🤖 : `, response.message.text);
  }
} catch (error) {
  reader.write(`Agent (error) 🤖 : `, FrameworkError.ensure(error).dump());
} finally {
  reader.close();
}
