/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyAgent } from "@/agents/types.js";
import { BaseAgent } from "@/agents/base.js";
import { Server } from "@/serve/server.js";
import express from "express";
import { Logger } from "@/logger/logger.js";
import { ChatCompletionAPI } from "./api.js";
import { ResponsesAPI } from "./responses_api.js";
import { ValueError } from "@/errors.js";

export type OpenAIAPIType = "chat-completion" | "responses";

const logger = Logger.root.child({
  name: "OpenAI server",
});

export class OpenAIServerConfig {
  host = "0.0.0.0";
  port = 9999;
  api: OpenAIAPIType = "chat-completion";
  apiKey?: string;
}

export interface OpenAIServerMetadata {
  name?: string;
  description?: string;
}

export class OpenAIServer extends Server<AnyAgent, AnyAgent, OpenAIServerConfig, OpenAIServerMetadata> {
  private ready = false;

  constructor(config: OpenAIServerConfig = new OpenAIServerConfig()) {
    super(config);
  }

  public register(input: AnyAgent, metadata?: OpenAIServerMetadata) {
    super.register(input, metadata);
    return this;
  }

  public async serve(): Promise<void> {
    if (this.members.length === 0) {
      throw new ValueError("No agents registered to the server.");
    }

    const app = express();

    const modelFactory = async (modelId: string): Promise<AnyAgent> => {
      // Find the first agent that matches the requested model ID by name.
      // If the client doesn't pass a specific name or it doesn't match, we fallback to the first member.
      const matchedMember = this.members.find(
        (m) => this.metadataByInput.get(m)?.name === modelId || m.meta.name === modelId
      );
      
      const member = matchedMember ?? this.members[0];
      if (!member) {
        throw new ValueError(`Model '${modelId}' not registered`);
      }
      return member;
    };

    const api =
      this.config.api === "responses"
        ? new ResponsesAPI(modelFactory, this.config.apiKey)
        : new ChatCompletionAPI(modelFactory, this.config.apiKey);

    // Mount the API under /v1 to simulate standard OpenAI structure
    app.use("/v1", api.router);
    // Also mount at root for direct access
    app.use("/", api.router);

    return new Promise((resolve, reject) => {
      const server = app.listen(this.config.port, this.config.host, () => {
        this.ready = true;
        logger.info(
          `OpenAI-compatible server started on http://${this.config.host}:${this.config.port}`,
        );
        logger.info(`Press Ctrl+C to stop the server`);
        resolve();
      });

      server.on("error", (error) => {
        logger.error(error, "Failed to start server");
        reject(error);
      });
    });
  }
}

const defaultAgentFactory = async (
  agent: AnyAgent,
  _?: OpenAIServerMetadata,
): Promise<AnyAgent> => {
  return agent;
};

// Assuming all agents can be wrapped this way as an interim abstraction
// Since BaseAgent is the base abstraction.
OpenAIServer.registerFactory(BaseAgent, defaultAgentFactory);
