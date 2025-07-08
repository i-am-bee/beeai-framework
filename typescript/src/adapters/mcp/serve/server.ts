/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ValueError } from "@/errors.js";
import { Server } from "@/serve/server.js";
import { ServerOptions } from "@modelcontextprotocol/sdk/server/index.js";
import {
  McpServer,
  PromptCallback,
  ReadResourceCallback,
  ReadResourceTemplateCallback,
  ResourceTemplate,
  ToolCallback,
} from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { AnyTool, Tool } from "@/tools/base.js";
import { runServer } from "./http_server.js";
import { ZodRawShape, ZodType } from "zod";
import { ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";

type MCPServerPrompt =
  | {
      type: "prompt";
      name: string;
      description: string;
      callback: PromptCallback;
    }
  | {
      type: "prompt";
      name: string;
      description: string;
      argsSchema: ZodRawShape;
      callback: PromptCallback<ZodRawShape>;
    };

type MCPServerResource =
  | {
      type: "resource";
      name: string;
      uri: string;
      callback: ReadResourceCallback;
    }
  | {
      type: "resource";
      name: string;
      template: ResourceTemplate;
      callback: ReadResourceTemplateCallback;
    };

interface MCPServerTool {
  type: "tool";
  name: string;
  description: string;
  paramsSchema: ZodRawShape | ToolAnnotations;
  callback: ToolCallback<ZodRawShape>;
}

type MCPServerEntry = MCPServerPrompt | MCPServerResource | MCPServerTool;

//  Configuration for the MCPServer.
export class MCPServerConfig {
  transport: "stdio" | "sse" = "stdio";
  hostname = "127.0.0.1";
  port = 3000;
  name = "MCP Server";
  version = "1.0.0";
  settings?: ServerOptions;

  constructor(partial?: Partial<MCPServerConfig>) {
    if (partial) {
      Object.assign(this, partial);
    }
  }
}

export class MCPServer extends Server<any, MCPServerEntry, MCPServerConfig> {
  protected server: McpServer;

  constructor(config?: MCPServerConfig) {
    super(config || new MCPServerConfig());
    this.server = new McpServer({
      name: this.config.name,
      version: this.config.version,
      ...this.config.settings,
    });
  }

  async serve() {
    for (const member of this.members) {
      const factory = this.getFactory(member);
      const entry = await factory(member);

      switch (entry.type) {
        case "tool":
          this.server.tool(entry.name, entry.description, entry.paramsSchema, entry.callback);
          break;
        case "prompt":
          if ("argsSchema" in entry) {
            this.server.prompt(entry.name, entry.description, entry.argsSchema, entry.callback);
          } else {
            this.server.prompt(entry.name, entry.description, entry.callback);
          }
          break;
        case "resource":
          if ("uri" in entry) {
            this.server.resource(entry.name, entry.uri, entry.callback);
          } else {
            this.server.resource(entry.name, entry.template, entry.callback);
          }
          break;
        default:
          throw new ValueError("Input type is not supported by this server.");
      }
    }

    if (this.config.transport === "sse") {
      runServer(this.server, this.config.hostname, this.config.port);
    } else {
      await this.server.connect(new StdioServerTransport());
    }
  }

  getFactory(member: any) {
    const factories = (this.constructor as typeof Server).factories;
    return !factories.has(member.constructor) && member instanceof Tool && factories.has(Tool)
      ? factories.get(Tool)!
      : super.getFactory(member);
  }
}

async function toolFactory(tool: AnyTool): Promise<MCPServerEntry> {
  const schema = await tool.inputSchema();
  if (!(schema instanceof ZodType)) {
    throw new ValueError("JsonSchema is not supported for MCP tools.");
  }
  const paramsSchema = schema.shape;
  return {
    type: "tool",
    name: tool.name,
    description: tool.description,
    paramsSchema: paramsSchema,
    callback: async (...args: Parameters<typeof tool.run>) => {
      const result = await tool.run(...args);
      return {
        content: [
          {
            type: "text",
            text: result.getTextContent(),
          },
        ],
      };
    },
  };
}

MCPServer.registerFactory(Tool, toolFactory);
