/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyAgent } from "@/agents/types.js";
import { Server } from "@/serve/server.js";
import {
  InMemoryTaskStore,
  TaskStore,
  AgentExecutor,
  DefaultRequestHandler,
  PushNotificationSender,
  ExecutionEventBusManager,
  PushNotificationStore,
} from "@a2a-js/sdk/server";
import { ReActAgentExecutor, ToolCallingAgentExecutor } from "./agent_executor.js";
import {
  AGENT_CARD_PATH,
  AgentCapabilities,
  AgentCard,
  AgentProvider,
  AgentSkill,
} from "@a2a-js/sdk";
import { A2AExpressApp } from "@a2a-js/sdk/server/express";
import express from "express";
import { ToolCallingAgent } from "@/agents/toolCalling/agent.js";
import { ValueError } from "@/errors.js";
import { Logger } from "@/logger/logger.js";
import { ReActAgent } from "@/agents/react/agent.js";

const logger = Logger.root.child({
  name: "A2A server",
});

export class A2AServerConfig {
  host = "0.0.0.0";
  port = 9999;
  protocol: "jsonrpc" = "jsonrpc" as const;
}

interface A2AServerMetadata {
  name?: string;
  description?: string;
  url?: string;
  version?: string;
  provider?: AgentProvider;
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  capabilities?: AgentCapabilities;
  skills?: AgentSkill[];
  taskStore?: TaskStore;
  eventBusManager?: ExecutionEventBusManager;
  pushNotificationStore?: PushNotificationStore;
  pushNotificationSender?: PushNotificationSender;
}

export class A2AServer extends Server<AnyAgent, AgentExecutor, A2AServerConfig, A2AServerMetadata> {
  constructor(config: A2AServerConfig = new A2AServerConfig()) {
    super(config);
  }

  public register(input: AnyAgent, metadata?: A2AServerMetadata) {
    if (this.members.length != 0) {
      throw new ValueError("A2AServer only supports one agent.");
    } else {
      super.register(input, metadata);
    }
    return this;
  }

  public registerMany(_: AnyAgent[]): this {
    throw new ValueError("RegisterMany is not implemented for A2AServer");
  }

  public async serve(): Promise<void> {
    if (this.members.length === 0) {
      throw new ValueError("No agents registered to the server.");
    }

    const member = this.members[0];
    const factory = this.getFactory(member);
    const config = this.metadataByInput.get(member) || {};
    const executor = await factory(member, config);
    const agentCard = this.createAgentCard(config, member);

    const taskStore: TaskStore = config.taskStore || new InMemoryTaskStore();

    const requestHandler = new DefaultRequestHandler(
      agentCard,
      taskStore,
      executor,
      config.eventBusManager,
      config.pushNotificationStore,
      config.pushNotificationSender,
    );

    const appBuilder = new A2AExpressApp(requestHandler);
    const expressApp = appBuilder.setupRoutes(express());

    expressApp.listen(this.config.port, this.config.host, () => {
      logger.info(
        `[${agentCard.name}] Server started on http://${this.config.host}:${this.config.port}`,
      );
      logger.info(
        `[${agentCard.name}] Agent Card: http://${this.config.host}:${this.config.port}/${AGENT_CARD_PATH}`,
      );
      logger.info(`[${agentCard.name}] Press Ctrl+C to stop the server`);
    });
  }

  private createAgentCard(config: A2AServerMetadata, agent: AnyAgent): AgentCard {
    return {
      name: config.name || agent.meta.name,
      description: config.description || agent.meta.description,
      url: config.url || `http://${this.config.host}:${this.config.port}`,
      version: config.version || "1.0.0",
      protocolVersion: "0.1.0",
      provider: config.provider,
      defaultInputModes: config.defaultInputModes || ["text"],
      defaultOutputModes: config.defaultOutputModes || ["text", "task-status"],
      capabilities: config.capabilities || {
        streaming: true,
        pushNotifications: false,
        stateTransitionHistory: false,
      },
      skills: config.skills || [
        {
          id: agent.meta.name,
          name: agent.meta.name,
          description: agent.meta.description,
          tags: [],
        },
      ],
      supportsAuthenticatedExtendedCard: false,
    };
  }
}

const toolCallingAgentFactory = async (
  agent: AnyAgent,
  _?: A2AServerMetadata,
): Promise<ToolCallingAgentExecutor> => {
  return new ToolCallingAgentExecutor(agent);
};

A2AServer.registerFactory(ToolCallingAgent, toolCallingAgentFactory);

const ReActAgentFactory = async (
  agent: AnyAgent,
  _?: A2AServerMetadata,
): Promise<ReActAgentExecutor> => {
  return new ReActAgentExecutor(agent);
};

A2AServer.registerFactory(ReActAgent, ReActAgentFactory);
