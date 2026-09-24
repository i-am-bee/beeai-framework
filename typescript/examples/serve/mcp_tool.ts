import { StringToolOutput, Tool, ToolEmitter, ToolInput } from "beeai-framework/tools/base";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { MCPServer, MCPServerConfig } from "beeai-framework/adapters/mcp/serve/server";
import { Emitter } from "beeai-framework/emitter/emitter";
import { z } from "zod";

export class ReverseTool extends Tool<StringToolOutput> {
  name = "ReverseTool";
  description = "A tool that reverses a word";

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "reverseTool"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      word: z.string(),
    });
  }

  protected async _run(input: ToolInput<this>): Promise<StringToolOutput> {
    return new StringToolOutput(input.word.split("").reverse().join(""));
  }
}

// create an MCP server with custom config, register ReverseTool and OpenMeteoTool and run it
const config = new MCPServerConfig({ transport: "streamable-http", port: 8001 });
const server = new MCPServer(config);
server.registerMany([new ReverseTool(), new OpenMeteoTool()]);
await server.serve();
