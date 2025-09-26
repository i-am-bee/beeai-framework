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
import { Callback, Emitter } from "@/emitter/emitter.js";
import { ToolCallingAgentCallbacks, ToolCallingAgentRunState } from "@/agents/toolCalling/types.js";
import { ReActAgentCallbacks } from "@/agents/react/types.js";
import { Message as FrameworkMessage } from "@/backend/message.js";

const logger = Logger.root.child({
  name: "A2A server",
});

export abstract class BaseA2AAgentExecutor implements AgentExecutor {
  protected readonly abortControllers = new Map<string, [string, AbortController]>();

  constructor(protected agent: AnyAgent) {}

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

    const agent = await this.agent.clone();
    agent.memory = await agent.memory.clone();
    agent.memory.reset();
    await agent.memory.addMany(
      (existingTask.history || [requestContext.userMessage]).map(
        convertA2AMessageToFrameworkMessage,
      ) || [],
    );

    try {
      // run the agent
      const response = await agent
        .run(
          {},
          {
            signal: abortController.signal,
          },
        )
        .observe(async (emitter) => {
          await this.process_events(emitter, requestContext, eventBus);
        });

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
      logger.error("Agent execution error:", { error });
      const errorMessage = error instanceof Error ? error.message : String(error);
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
            parts: [{ kind: "text", text: `Agent error: ${errorMessage}` }],
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

  async process_events(
    _emitter: Emitter,
    _requestContext: RequestContext,
    _eventBus: ExecutionEventBus,
  ): Promise<void> {
    return;
  }

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
  async process_events(
    emitter: Emitter<ToolCallingAgentCallbacks>,
    requestContext: RequestContext,
    eventBus: ExecutionEventBus,
  ): Promise<void> {
    let lastMsg: FrameworkMessage | undefined = undefined;

    const processEvent: Callback<{ state: ToolCallingAgentRunState }> = async ({ state }) => {
      const messages = state.memory.messages;
      if (lastMsg === undefined) {
        lastMsg = messages.at(-1);
      }

      const index = lastMsg ? messages.lastIndexOf(lastMsg) : -1;
      for (const message of messages.slice(index + 1)) {
        const update: TaskStatusUpdateEvent = {
          kind: "status-update",
          taskId: requestContext.taskId,
          contextId: requestContext.contextId,
          status: {
            state: "working",
            message: {
              kind: "message",
              role: "agent",
              messageId: uuidv4(),
              parts: [{ kind: "text", text: JSON.stringify(message.toPlain()) }],
              taskId: requestContext.taskId,
              contextId: requestContext.contextId,
            },
            timestamp: new Date().toISOString(),
          },
          final: false,
        };
        eventBus.publish(update);
        lastMsg = message;
      }
    };
    emitter.on("start", processEvent);
    emitter.on("success", processEvent);
  }
}

export class ReActAgentExecutor extends BaseA2AAgentExecutor {
  async process_events(
    emitter: Emitter<ReActAgentCallbacks>,
    requestContext: RequestContext,
    eventBus: ExecutionEventBus,
  ): Promise<void> {
    emitter.on("update", async ({ update }) => {
      const updateEvent: TaskStatusUpdateEvent = {
        kind: "status-update",
        taskId: requestContext.taskId,
        contextId: requestContext.contextId,
        status: {
          state: "working",
          message: {
            kind: "message",
            role: "agent",
            messageId: uuidv4(),
            parts: [{ kind: "text", text: update.value }],
            taskId: requestContext.taskId,
            contextId: requestContext.contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: false,
      };
      eventBus.publish(updateEvent);
    });
  }
}
