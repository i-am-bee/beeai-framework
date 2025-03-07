import "dotenv/config.js";
import { Client as MCPClient } from "@i-am-bee/acp-sdk/client/index.js";
import { SSEClientTransport } from "@i-am-bee/acp-sdk/client/sse.js";
import { RemoteAgent } from "beeai-framework/agents/experimental/remote/agent";
import { createConsoleReader } from "examples/helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";

const agentName = "literature-review";

const instance = new RemoteAgent({
  client: new MCPClient({
    name: "example-remote-agent",
    version: "1.0.0",
  }),
  transport: new SSEClientTransport(new URL("/mcp/sse", "http://127.0.0.1:8333")),
  agent: agentName,
});

const reader = createConsoleReader();

try {
  for await (const { prompt } of reader) {
    const result = await instance.run({ prompt: JSON.parse(prompt) }).observe((emitter) => {
      emitter.on("update", (data) => {
        reader.write(`Agent (received progress) 🤖 : `, data.output);
      });
    });

    reader.write(`Agent (${agentName}) 🤖 : `, result.message.text);
  }
} catch (error) {
  reader.write("Agent (error)  🤖", FrameworkError.ensure(error).dump());
}
