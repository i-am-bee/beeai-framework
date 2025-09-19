import "dotenv/config.js";
import { A2AAgent } from "beeai-framework/adapters/a2a/agents/agent";
import { createConsoleReader } from "examples/helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";

const agent = new A2AAgent({
  url: "http://127.0.0.1:9999",
  memory: new TokenMemory(),
});

const reader = createConsoleReader();

try {
  for await (const { prompt } of reader) {
    const result = await agent.run({ input: prompt }).observe((emitter) => {
      emitter.on("update", (data) => {
        reader.write(`Agent (received progress) 🤖 : `, JSON.stringify(data.value, null, 2));
      });
      emitter.on("error", (data) => {
        reader.write(`Agent (error) 🤖 : `, data.message);
      });
    });

    reader.write(`Agent 🤖 : `, result.result.text);
  }
} catch (error) {
  reader.write("Agent (error) 🤖", FrameworkError.ensure(error).dump());
}
