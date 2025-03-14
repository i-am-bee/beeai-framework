import "dotenv/config.js";
import { RemoteAgent } from "beeai-framework/agents/experimental/remote/agent";
import { createConsoleReader } from "examples/helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";

const agentName = "chat";

const instance = RemoteAgent.createSSEAgent("http://127.0.0.1:8333/mcp/sse", agentName);

const reader = createConsoleReader();

try {
  for await (const { prompt } of reader) {
    const result = await instance
      .run({
        prompt: {
          messages: [{ role: "user", content: prompt }],
          config: { tools: ["weather", "search", "wikipedia"] },
        },
      })
      .observe((emitter) => {
        emitter.on("update", (data) => {
          if (JSON.parse(data.output)?.["logs"]) {
            reader.write(
              `Agent (received progress) 🤖 : `,
              JSON.parse(data.output)?.["logs"]?.[0]?.["message"],
            );
          }
        });
      });

    reader.write(`Agent (${agentName}) 🤖 : `, result.message.text);
  }
} catch (error) {
  reader.write("Agent (error)  🤖", FrameworkError.ensure(error).dump());
}
