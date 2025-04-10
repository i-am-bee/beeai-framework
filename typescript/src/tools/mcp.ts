/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { BaseToolRunOptions, ToolEmitter, ToolInput, JSONToolOutput, Tool } from "@/tools/base.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext } from "@/context.js";
import { Client as MCPClient } from "@modelcontextprotocol/sdk/client/index.js";
import { ListToolsResult } from "@modelcontextprotocol/sdk/types.js";
import { SchemaObject } from "ajv";
import { paginate } from "@/internals/helpers/paginate.js";

export interface MCPToolInput {
  client: MCPClient;
  tool: ListToolsResult["tools"][number];
}

export class MCPToolOutput extends JSONToolOutput<any> {}

export class MCPTool extends Tool<MCPToolOutput> {
  public readonly name: string;
  public readonly description: string;
  public readonly emitter: ToolEmitter<ToolInput<this>, MCPToolOutput>;

  public readonly client: MCPClient;
  private readonly tool: ListToolsResult["tools"][number];

  public constructor({ client, tool, ...options }: MCPToolInput) {
    super(options);
    this.client = client;
    this.tool = tool;
    this.name = tool.name;
    this.description =
      tool.description ?? "No available description, use the tool based on its name and schema.";
    this.emitter = Emitter.root.child({
      namespace: ["tool", "mcp", this.name],
      creator: this,
    });
  }

  public inputSchema() {
    return this.tool.inputSchema as SchemaObject;
  }

  protected async _run(
    input: ToolInput<this>,
    _options: BaseToolRunOptions,
    run: GetRunContext<typeof this>,
  ) {
    const result = await this.client.callTool({ name: this.name, arguments: input }, undefined, {
      signal: run.signal,
    });
    return new MCPToolOutput(result);
  }

  public static async fromClient(client: MCPClient): Promise<MCPTool[]> {
    const tools = await paginate({
      size: Infinity,
      handler: async ({ cursor }: { cursor?: string }) => {
        const { tools, nextCursor } = await client.listTools({ cursor });
        return { data: tools, nextCursor } as const;
      },
    });
    return tools.map((tool) => new MCPTool({ client, tool }));
  }
}
