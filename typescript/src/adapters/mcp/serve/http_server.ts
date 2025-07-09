/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Taken from: https://github.com/modelcontextprotocol/typescript-sdk/blob/main/src/examples/server/sseAndStreamableHttpCompatibleServer.ts

import express, { Request, Response } from "express";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { InMemoryEventStore } from "./in_memory_store.js";
import { randomUUID } from "node:crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Logger } from "@/logger/logger.js";

const logger = Logger.root.child({
  name: "MCP HTTP server",
});

export function runServer(server: McpServer, hostname = "127.0.0.1", port = 3000) {
  // Create Express application
  const app = express();
  app.use(express.json());

  // Store transports by session ID
  const transports: Record<string, StreamableHTTPServerTransport | SSEServerTransport> = {};

  //=============================================================================
  // STREAMABLE HTTP TRANSPORT (PROTOCOL VERSION 2025-03-26)
  //=============================================================================

  // Handle all MCP Streamable HTTP requests (GET, POST, DELETE) on a single endpoint
  app.all("/mcp", async (req: Request, res: Response) => {
    logger.debug(`Received ${req.method} request to /mcp`);

    try {
      // Check for existing session ID
      const sessionId = req.headers["mcp-session-id"] as string | undefined;
      let transport: StreamableHTTPServerTransport;

      if (sessionId && transports[sessionId]) {
        // Check if the transport is of the correct type
        const existingTransport = transports[sessionId];
        if (existingTransport instanceof StreamableHTTPServerTransport) {
          // Reuse existing transport
          transport = existingTransport;
        } else {
          // Transport exists but is not a StreamableHTTPServerTransport (could be SSEServerTransport)
          res.status(400).json({
            jsonrpc: "2.0",
            error: {
              code: -32000,
              message: "Bad Request: Session exists but uses a different transport protocol",
            },
            id: null,
          });
          return;
        }
      } else if (!sessionId && req.method === "POST" && isInitializeRequest(req.body)) {
        const eventStore = new InMemoryEventStore();
        transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomUUID(),
          eventStore, // Enable resumability
          onsessioninitialized: (sessionId) => {
            // Store the transport by session ID when session is initialized
            logger.debug(`StreamableHTTP session initialized with ID: ${sessionId}`);
            transports[sessionId] = transport;
          },
        });

        // Set up onclose handler to clean up transport when closed
        transport.onclose = () => {
          const sid = transport.sessionId;
          if (sid && transports[sid]) {
            logger.debug(`Transport closed for session ${sid}, removing from transports map`);
            // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
            delete transports[sid];
          }
        };

        // Connect the transport to the MCP server
        await server.connect(transport);
      } else {
        // Invalid request - no session ID or not initialization request
        res.status(400).json({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Bad Request: No valid session ID provided",
          },
          id: null,
        });
        return;
      }

      // Handle the request with the transport
      await transport.handleRequest(req, res, req.body);
    } catch (error) {
      logger.error("Error handling MCP request:", error);
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: "2.0",
          error: {
            code: -32603,
            message: "Internal server error",
          },
          id: null,
        });
      }
    }
  });

  //=============================================================================
  // DEPRECATED HTTP+SSE TRANSPORT (PROTOCOL VERSION 2024-11-05)
  //=============================================================================

  app.get("/sse", async (req: Request, res: Response) => {
    logger.info("Received GET request to /sse (deprecated SSE transport)");
    const transport = new SSEServerTransport("/messages", res);
    transports[transport.sessionId] = transport;
    res.on("close", () => {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete transports[transport.sessionId];
    });
    await server.connect(transport);
  });

  app.post("/messages", async (req: Request, res: Response) => {
    const sessionId = req.query.sessionId as string;
    let transport: SSEServerTransport;
    const existingTransport = transports[sessionId];
    if (existingTransport instanceof SSEServerTransport) {
      // Reuse existing transport
      transport = existingTransport;
    } else {
      // Transport exists but is not a SSEServerTransport (could be StreamableHTTPServerTransport)
      res.status(400).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: Session exists but uses a different transport protocol",
        },
        id: null,
      });
      return;
    }
    if (transport) {
      await transport.handlePostMessage(req, res, req.body);
    } else {
      res.status(400).send("No transport found for sessionId");
    }
  });

  // Start the server
  app.listen(port, hostname, () => {
    logger.info(`Backwards compatible MCP server listening on port ${hostname}:${port}`);
    logger.debug(`
    ==============================================
    SUPPORTED TRANSPORT OPTIONS:

    1. Streamable Http(Protocol version: 2025-03-26)
    Endpoint: /mcp
    Methods: GET, POST, DELETE
    Usage: 
        - Initialize with POST to /mcp
        - Establish SSE stream with GET to /mcp
        - Send requests with POST to /mcp
        - Terminate session with DELETE to /mcp

    2. Http + SSE (Protocol version: 2024-11-05)
    Endpoints: /sse (GET) and /messages (POST)
    Usage:
        - Establish SSE stream with GET to /sse
        - Send requests with POST to /messages?sessionId=<id>
    ==============================================
    `);
  });

  // Handle server shutdown
  process.on("SIGINT", async () => {
    logger.info("Shutting down server...");

    // Close all active transports to properly clean up resources
    for (const sessionId in transports) {
      try {
        logger.debug(`Closing transport for session ${sessionId}`);
        await transports[sessionId].close();
        // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
        delete transports[sessionId];
      } catch (error) {
        logger.error(`Error closing transport for session ${sessionId}:`, error);
      }
    }
    logger.debug("Server shutdown complete");
    process.exit(0);
  });
}
