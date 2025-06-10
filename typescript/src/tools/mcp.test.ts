/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { MCPTool } from "./mcp.js";
import { entries } from "remeda";
import { zodToJsonSchema } from "zod-to-json-schema";
import { z } from "zod";

const abInputSchema = z.object({ a: z.number(), b: z.number() });
const toolDefinition = {
  add: {
    description: "Adds two numbers",
    inputSchema: zodToJsonSchema(abInputSchema),
    exampleArguments: { a: 2, b: 3 },
    handler: vi.fn(({ a, b }: z.input<typeof abInputSchema>) => a + b),
  },
  multiply: {
    description: "Multiplies two numbers",
    inputSchema: zodToJsonSchema(abInputSchema),
    exampleArguments: { a: 2, b: 3 },
    handler: vi.fn(({ a, b }: z.input<typeof abInputSchema>) => a * b),
  },
} as const;

describe("MCPTool", () => {
  let server: Server;
  let client: Client;

  beforeEach(async () => {
    server = new Server(
      {
        name: "test-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      },
    );
    server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: entries(toolDefinition).map(([name, { description, inputSchema }]) => ({
          name,
          description,
          inputSchema,
        })),
      };
    });
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const tool = toolDefinition[request.params.name as keyof typeof toolDefinition];
      if (!tool) {
        throw new Error("Tool not found");
      }
      // Arguments are assumed to be valid in this mock
      return {
        content: [{ type: "text", text: String(tool.handler(request.params.arguments as any)) }],
      };
    });

    client = new Client(
      {
        name: "test-client",
        version: "1.0.0",
      },
      {
        capabilities: {},
      },
    );

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
  });

  it("should list tools", async () => {
    const tools = await MCPTool.fromClient(client);
    expect(tools.length).toBe(Object.keys(toolDefinition).length);
  });

  it("should call a tool", async () => {
    const tool = (await MCPTool.fromClient(client))[0];
    const toolDef = toolDefinition[tool.name as keyof typeof toolDefinition];
    await tool.run(toolDef.exampleArguments);
    expect(toolDef.handler).toBeCalled();
  });

  afterEach(async () => {
    await client.close();
    await server.close();

    vi.clearAllMocks();
  });
});
