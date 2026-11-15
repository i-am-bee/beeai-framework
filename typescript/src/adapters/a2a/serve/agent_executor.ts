/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnyAgent } from "@/agents/types.js";
import { AgentEvent, AgentExecutor, RequestContext, ExecutionEventBus } from "@a2a-js/sdk/server";
import { Message, Role, Task, TaskState, TaskStatusUpdateEvent } from "@a2a-js/sdk";
import { Logger } from "@/logger/logger.js";
import {
  convertA2AMessageToFrameworkMessage,
  createA2AMessage,
  createA2APart,
} from "@/adapters/a2a/agents/utils.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { ToolCallingAgentCallbacks, ToolCallingAgentRunState } from "@/agents/toolCalling/types.js";
import { ReActAgentCallbacks } from "@/agents/react/types.js";
import { Message as FrameworkMessage } from "@/backend/message.js";

const logger = Logger.root.child({
  name: "A2A server",
});

function createStatusUpdate(
  taskId: string,
  contextId: string,
  state: TaskState,
  message?: Message,
): TaskStatusUpdateEvent {
  return {
    taskId,
    contextId,
    status: { state, message, timestamp: new Date().toISOString() },
    metadata: undefined,
  };
}

function createAgentMessage(text: string, taskId: string, contextId: string): Message {
  return createA2AMessage({
    role: Role.ROLE_AGENT,
    parts: [createA2APart({ $case: "text", value: text })],
    taskId,
    contextId,
  });
}

export abstract class BaseA2AAgentExecutor implements AgentExecutor {
  protected readonly abortControllers = new Map<string, [string, AbortController]>();

  constructor(protected agent: AnyAgent) {}

  async execute(requestContext: RequestContext, eventBus: ExecutionEventBus): Promise<void> {
    const userMessage = requestContext.userMessage;
    // a2a-js 1.x checks that the events carry the task id it assigned to the request
    const existingTask: Task = requestContext.task || {
      id: requestContext.taskId,
      contextId: requestContext.contextId,
      status: {
        state: TaskState.TASK_STATE_SUBMITTED,
        message: undefined,
        timestamp: new Date().toISOString(),
      },
      artifacts: [],
      history: [userMessage],
      metadata: userMessage.metadata,
    };

    const taskId = existingTask.id;
    const contextId = existingTask.contextId;

    // a2a-js 1.x requires a task or a message as the first event, also on follow-up turns
    eventBus.publish(AgentEvent.task(existingTask));

    eventBus.publish(
      AgentEvent.statusUpdate(createStatusUpdate(taskId, contextId, TaskState.TASK_STATE_WORKING)),
    );

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

      const agentMessage = createAgentMessage(response.result.text, taskId, contextId);

      // Publish completed status update with agent message, which a2a-js adds to the task history
      // (a second task event in the same stream is rejected since 1.x)
      eventBus.publish(
        AgentEvent.statusUpdate(
          createStatusUpdate(taskId, contextId, TaskState.TASK_STATE_COMPLETED, agentMessage),
        ),
      );

      eventBus.finished();
    } catch (error) {
      logger.error(error, "Agent execution error");
      const errorMessage = error instanceof Error ? error.message : String(error);
      // Publish failed status update
      const failedMessage = createAgentMessage(`Agent error: ${errorMessage}`, taskId, contextId);
      eventBus.publish(
        AgentEvent.statusUpdate(
          createStatusUpdate(taskId, contextId, TaskState.TASK_STATE_FAILED, failedMessage),
        ),
      );
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
      eventBus.publish(
        AgentEvent.statusUpdate(
          createStatusUpdate(taskId, contextId, TaskState.TASK_STATE_CANCELED),
        ),
      );
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
        const { taskId, contextId } = requestContext;
        const text = JSON.stringify(message.toPlain());
        eventBus.publish(
          AgentEvent.statusUpdate(
            createStatusUpdate(
              taskId,
              contextId,
              TaskState.TASK_STATE_WORKING,
              createAgentMessage(text, taskId, contextId),
            ),
          ),
        );
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
      const { taskId, contextId } = requestContext;
      eventBus.publish(
        AgentEvent.statusUpdate(
          createStatusUpdate(
            taskId,
            contextId,
            TaskState.TASK_STATE_WORKING,
            createAgentMessage(update.value, taskId, contextId),
          ),
        ),
      );
    });
  }
}
