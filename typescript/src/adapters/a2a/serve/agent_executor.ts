/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { v4 as uuidv4 } from "uuid";

import { AnyAgent } from "@/agents/types.js";
import { AgentExecutor, RequestContext, ExecutionEventBus } from "@a2a-js/sdk/server";
import { Message, TaskStatusUpdateEvent } from "@a2a-js/sdk";
import { Logger } from "@/logger/logger.js";
import { convertA2AMessageToFrameworkMessage } from "@/adapters/a2a/agents/utils.js";

const logger = Logger.root.child({
  name: "A2A server",
});

export abstract class BaseA2AAgentExecutor implements AgentExecutor {
  protected readonly abortControllers = new Map<string, [string, AbortController]>();

  constructor(protected agent: AnyAgent) {}

  abstract execute(requestContext: RequestContext, eventBus: ExecutionEventBus): Promise<void>;

  public cancelTask = async (taskId: string, eventBus: ExecutionEventBus): Promise<void> => {
    if (this.abortControllers.has(taskId)) {
      const [contextId, abortController] = this.abortControllers.get(taskId)!;
      const cancelledUpdate: TaskStatusUpdateEvent = {
        kind: "status-update",
        taskId: taskId,
        contextId: contextId,
        status: {
          state: "canceled",
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(cancelledUpdate);
      abortController.abort();
    }
  };
}

export class ToolCallingAgentExecutor extends BaseA2AAgentExecutor {
  async execute(requestContext: RequestContext, eventBus: ExecutionEventBus): Promise<void> {
    const userMessage = requestContext.userMessage;
    const existingTask = requestContext.task || {
      kind: "task",
      id: uuidv4(),
      contextId: userMessage.contextId || uuidv4(),
      status: {
        state: "submitted",
        timestamp: new Date().toISOString(),
      },
      history: [userMessage],
      metadata: userMessage.metadata,
    };

    const taskId = existingTask.id;
    const contextId = existingTask.contextId;

    // Publish initial Task event if it's a new task
    eventBus.publish(existingTask);

    // Publish "working" status update
    const workingStatusUpdate: TaskStatusUpdateEvent = {
      kind: "status-update",
      taskId: taskId,
      contextId: contextId,
      status: {
        state: "working",
        timestamp: new Date().toISOString(),
      },
      final: false,
    };
    eventBus.publish(workingStatusUpdate);

    const abortController = new AbortController();
    this.abortControllers.set(taskId, [contextId, abortController]);

    try {
      // run the agent
      const response = await this.agent.run(
        convertA2AMessageToFrameworkMessage(requestContext.userMessage),
        {
          signal: abortController.signal,
        },
      );

      const agentMessage: Message = {
        kind: "message",
        role: "agent",
        messageId: uuidv4(),
        parts: [{ kind: "text", text: response.result.text }],
        taskId: taskId,
        contextId: contextId,
      };

      // Append agent message to task history
      if (!existingTask.history) {
        existingTask.history = [];
      }
      existingTask.history.push(agentMessage);
      eventBus.publish(existingTask);

      // Publish completed status update with agent message
      const finalUpdate: TaskStatusUpdateEvent = {
        kind: "status-update",
        taskId: taskId,
        contextId: contextId,
        status: {
          state: "completed",
          message: agentMessage,
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(finalUpdate);

      eventBus.finished();
    } catch (error) {
      logger.error("Agent execution error:", error);
      // Publish failed status update
      const errorUpdate: TaskStatusUpdateEvent = {
        kind: "status-update",
        taskId: taskId,
        contextId: contextId,
        status: {
          state: "failed",
          message: {
            kind: "message",
            role: "agent",
            messageId: uuidv4(),
            parts: [{ kind: "text", text: `Agent error: ${error.message}` }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(errorUpdate);
      eventBus.finished();
    }
  }
}
