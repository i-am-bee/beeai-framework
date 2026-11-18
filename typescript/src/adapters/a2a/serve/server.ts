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
  A2A_PROTOCOL_VERSION,
  AGENT_CARD_PATH,
  AgentCapabilities,
  AgentCard,
  AgentInterface,
  AgentProvider,
  AgentSkill,
} from "@a2a-js/sdk";
import { A2A_LEGACY_PROTOCOL_VERSION } from "@a2a-js/sdk/compat/v0_3";
import { agentCardHandler, jsonRpcHandler, UserBuilder } from "@a2a-js/sdk/server/express";
import express, { Express, Request, Response } from "express";
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
  // also serve clients still on the 0.3 protocol, such as an A2AAgent on @a2a-js/sdk 0.3
  enableV03Compat = true;
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
  healthCheck?: () => Promise<boolean>;
}

export class A2AServer extends Server<AnyAgent, AgentExecutor, A2AServerConfig, A2AServerMetadata> {
  private ready = false;

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
    const { app: expressApp, agentCard } = await this.createApp();

    expressApp.listen(this.config.port, this.config.host, () => {
      this.ready = true;
      logger.info(
        `[${agentCard.name}] Server started on http://${this.config.host}:${this.config.port}`,
      );
      logger.info(
        `[${agentCard.name}] Agent Card: http://${this.config.host}:${this.config.port}/${AGENT_CARD_PATH}`,
      );
      logger.info(`[${agentCard.name}] Press Ctrl+C to stop the server`);
    });
  }

  /**
   * Builds the Express app that `serve()` listens with. When it is served by other means, `/health`
   * reports ready only through the `healthCheck` given in the metadata.
   */
  public async createApp(): Promise<{ app: Express; agentCard: AgentCard }> {
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

    const legacyCompat = { enabled: this.config.enableV03Compat };
    const expressApp = express();

    expressApp.get("/health", async (_req: Request, res: Response) => {
      const timeout = (ms: number) =>
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error("timeout")), ms));

      try {
        const healthPromise = config.healthCheck
          ? config.healthCheck()
          : Promise.resolve(this.ready);
        const healthy = await Promise.race([healthPromise, timeout(2000)]);
        res.status(healthy ? 200 : 503).send(healthy ? "ok" : "not ready");
      } catch {
        res.status(503).send("not ready");
      }
    });
    expressApp.use(
      `/${AGENT_CARD_PATH}`,
      agentCardHandler({ agentCardProvider: requestHandler, legacyCompat }),
    );
    expressApp.use(
      "/",
      jsonRpcHandler({ requestHandler, userBuilder: UserBuilder.noAuthentication, legacyCompat }),
    );

    return { app: expressApp, agentCard };
  }

  private createAgentCard(config: A2AServerMetadata, agent: AnyAgent): AgentCard {
    const url = config.url || `http://${this.config.host}:${this.config.port}`;
    const supportedInterfaces: AgentInterface[] = [
      { url, protocolBinding: "JSONRPC", tenant: "", protocolVersion: A2A_PROTOCOL_VERSION },
    ];
    if (this.config.enableV03Compat) {
      // a2a-js serves 0.3 requests only when the card also lists the 0.3 interface
      supportedInterfaces.push({
        ...supportedInterfaces[0],
        protocolVersion: A2A_LEGACY_PROTOCOL_VERSION,
      });
    }

    return {
      name: config.name || agent.meta.name,
      description: config.description || agent.meta.description,
      supportedInterfaces,
      version: config.version || "1.0.0",
      provider: config.provider,
      defaultInputModes: config.defaultInputModes || ["text"],
      defaultOutputModes: config.defaultOutputModes || ["text", "task-status"],
      capabilities: config.capabilities || {
        streaming: true,
        pushNotifications: false,
        extensions: [],
      },
      skills: config.skills || [
        {
          id: agent.meta.name,
          name: agent.meta.name,
          description: agent.meta.description,
          tags: [],
          examples: [],
          inputModes: [],
          outputModes: [],
          securityRequirements: [],
        },
      ],
      securitySchemes: {},
      securityRequirements: [],
      signatures: [],
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
