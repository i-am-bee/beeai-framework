import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { MCPTool } from "beeai-framework/tools/mcp";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// Create MCP Client
const client = new Client(
  {
    name: "test-client",
    version: "1.0.0",
  },
  {
    capabilities: {},
  },
);

// Connect the client to any MCP server with tools capablity
await client.connect(
  new StdioClientTransport({
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-everything"],
  }),
);

try {
  // Server usually supports several tools, use the factory for automatic discovery
  const tools = await MCPTool.fromClient(client);
  const agent = new ReActAgent({
    llm: new OllamaChatModel("llama3.1"),
    memory: new UnconstrainedMemory(),
    tools,
  });
  // @modelcontextprotocol/server-everything contains "add" tool
  await agent.run({ prompt: "Find out how much is 4 + 7" }).observe((emitter) => {
    emitter.on("update", async ({ data, update, meta }) => {
      console.log(`Agent (${update.key}) 🤖 : `, update.value);
    });
  });
} finally {
  // Close the MCP connection
  await client.close();
}
