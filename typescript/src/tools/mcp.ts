/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseToolRunOptions,
  ToolEmitter,
  ToolInput,
  JSONToolOutput,
  Tool,
  ToolError,
} from "@/tools/base.js";
import { Emitter } from "@/emitter/emitter.js";
import { GetRunContext } from "@/context.js";
import { Client as MCPClient } from "@modelcontextprotocol/sdk/client/index.js";
import { ListToolsResult } from "@modelcontextprotocol/sdk/types.js";
import { SchemaObject } from "ajv";
import { paginate } from "@/internals/helpers/paginate.js";
import { isPlainObject, toCamelCase } from "remeda";

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
      namespace: ["tool", "mcp", toCamelCase(this.name)],
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
    if (result.isError) {
      const errorContext = result._meta?.error_context;
      throw new ToolError(MCPTool.formatError(result), [], {
        context: isPlainObject(errorContext) ? errorContext : undefined,
      });
    }
    return new MCPToolOutput(result);
  }

  protected static formatError(result: Awaited<ReturnType<MCPClient["callTool"]>>): string {
    if (result.structuredContent !== undefined) {
      return JSON.stringify(result.structuredContent, null, 4);
    }
    const content = Array.isArray(result.content) ? result.content : [];
    const text = content
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("\n");
    return text || JSON.stringify(content, null, 4);
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
