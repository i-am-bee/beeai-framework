/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import express, { Request, Response, Router } from "express";
import { v4 as uuidv4 } from "uuid";
import { AnyAgent } from "@/agents/types.js";
import { Logger } from "@/logger/logger.js";
import { SystemMessage, UserMessage, Message } from "@/backend/message.js";
import { openaiInputToBeeAIMessage } from "./responses_utils.js";
import {
  ResponsesRequestBody,
  ResponsesRequestInputMessage,
  ResponsesResponse,
  ResponsesMessageOutput,
  ResponsesMessageContent,
  ResponsesStreamResponseCreated,
  ResponsesStreamResponseInProgress,
  ResponsesStreamResponseCompleted,
  ResponsesStreamOutputItemAdded,
  ResponsesStreamOutputItemDone,
  ResponsesStreamContentPartAdded,
  ResponsesStreamContentPartDone,
  ResponsesStreamOutputTextDelta,
  ResponsesStreamOutputTextDone,
  ResponsesStreamPartOutputText,
  ResponsesStreamError,
} from "./responses_types.js";

const logger = Logger.root.child({
  name: "OpenAI Responses API",
});

export class ResponsesAPI {
  public readonly router: Router;

  constructor(
    private readonly modelFactory: (modelId: string) => Promise<AnyAgent>,
    private readonly apiKey?: string,
  ) {
    this.router = express.Router();
    this.router.use(express.json());

    this.router.post("/responses", this.handler.bind(this));
  }

  private async handler(req: Request, res: Response) {
    const requestBody = req.body as ResponsesRequestBody;
    logger.debug(`Received request: ${JSON.stringify(requestBody)}`);

    if (this.apiKey) {
      const authHeader = req.headers.authorization;
      const token = Array.isArray(authHeader) ? authHeader[0] : authHeader;
      if (!token || token.replace(/^Bearer\s+/i, "") !== this.apiKey) {
        res.status(401).json({ detail: "Missing or invalid API key" });
        return;
      }
    }

    try {
      const messages: Message[] = [];

      // Convert instructions to SystemMessage if present
      if (requestBody.instructions) {
        messages.push(new SystemMessage(requestBody.instructions));
      }

      // Convert input to BeeAI messages
      if (typeof requestBody.input === "string") {
        messages.push(new UserMessage(requestBody.input));
      } else {
        for (const msg of requestBody.input as ResponsesRequestInputMessage[]) {
          messages.push(openaiInputToBeeAIMessage(msg));
        }
      }

      const agent = await this.modelFactory(requestBody.model);

      // We clone the agent to avoid mutating the registered instance
      const clonedAgent = await agent.clone();
      clonedAgent.memory = await clonedAgent.memory.clone();
      clonedAgent.memory.reset();
      await clonedAgent.memory.addMany(messages);

      const responseId = `resp_${uuidv4()}`;

      if (requestBody.stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        let sequenceNumber = 0;
        const outputItemId = `msg_${uuidv4()}`;
        let accumulatedText = "";

        const sendEvent = (eventType: string, data: { sequence_number?: number }) => {
          sequenceNumber++;
          data.sequence_number = sequenceNumber;
          if (res.writableEnded || res.destroyed) {
            return;
          }
          try {
            res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
          } catch (writeErr) {
            logger.error(writeErr, `Failed to write event ${eventType} to response`);
          }
        };

        // response.created
        const createdResponse: ResponsesResponse = {
          id: responseId,
          object: "response",
          created: Math.floor(Date.now() / 1000),
          status: "in_progress",
          model: requestBody.model,
        };

        const createdEvent: ResponsesStreamResponseCreated = {
          type: "response.created",
          response: createdResponse,
        };
        sendEvent("response.created", createdEvent);

        // response.in_progress
        const inProgressEvent: ResponsesStreamResponseInProgress = {
          type: "response.in_progress",
          response: { ...createdResponse },
        };
        sendEvent("response.in_progress", inProgressEvent);

        // response.output_item.added
        const outputItem: ResponsesMessageOutput = {
          type: "message",
          id: outputItemId,
          status: "in_progress",
          role: "assistant",
          content: [],
        };
        const outputItemAddedEvent: ResponsesStreamOutputItemAdded = {
          type: "response.output_item.added",
          item: outputItem,
          output_index: 0,
        };
        sendEvent("response.output_item.added", outputItemAddedEvent);

        // response.content_part.added
        const emptyPart: ResponsesStreamPartOutputText = {
          text: "",
          type: "output_text",
          annotations: [],
        };
        const contentPartAddedEvent: ResponsesStreamContentPartAdded = {
          type: "response.content_part.added",
          content_index: 0,
          item_id: outputItemId,
          output_index: 0,
          part: emptyPart,
        };
        sendEvent("response.content_part.added", contentPartAddedEvent);

        // Abort the agent run if the client disconnects early
        const controller = new AbortController();
        req.on("close", () => {
          controller.abort();
        });

        try {
          // Use .observe() to subscribe to the run-scoped emitter — the correct
          // BeeAI pattern for intercepting agent events during a run.
          await clonedAgent
            .run({ prompt: null }, { signal: controller.signal })
            .observe((emitter) => {
              // Match all events — required because the emitter is typed as
              // Emitter<unknown> at the AnyAgent abstraction. We emit text deltas
              // from "update" (ReActAgent incremental tokens) and from "success"
              // (ToolCallingAgent emits only start/success, with the final answer
              // in state.result). This keeps streaming working for both agents.
              emitter.match("*.*", async (data: any, event) => {
                let delta: string | undefined;
                if (event.name === "update") {
                  delta = data?.update?.value;
                } else if (event.name === "success") {
                  delta = data?.state?.result?.text;
                }
                if (!delta || res.writableEnded || res.destroyed) {
                  return;
                }
                accumulatedText += delta;

                const deltaEvent: ResponsesStreamOutputTextDelta = {
                  type: "response.output_text.delta",
                  content_index: 0,
                  delta,
                  item_id: outputItemId,
                  output_index: 0,
                };
                sendEvent("response.output_text.delta", deltaEvent);
              });
            });

          // response.output_text.done
          const textDoneEvent: ResponsesStreamOutputTextDone = {
            type: "response.output_text.done",
            content_index: 0,
            text: accumulatedText,
            item_id: outputItemId,
            output_index: 0,
          };
          sendEvent("response.output_text.done", textDoneEvent);

          // response.content_part.done
          const finalPart: ResponsesStreamPartOutputText = {
            text: accumulatedText,
            type: "output_text",
            annotations: [],
          };
          const contentPartDoneEvent: ResponsesStreamContentPartDone = {
            type: "response.content_part.done",
            content_index: 0,
            item_id: outputItemId,
            output_index: 0,
            part: finalPart,
          };
          sendEvent("response.content_part.done", contentPartDoneEvent);

          // response.output_item.done
          const completedOutputItem: ResponsesMessageOutput = {
            type: "message",
            id: outputItemId,
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: accumulatedText }],
          };
          const outputItemDoneEvent: ResponsesStreamOutputItemDone = {
            type: "response.output_item.done",
            item: completedOutputItem,
            output_index: 0,
          };
          sendEvent("response.output_item.done", outputItemDoneEvent);

          // response.completed
          const completedResponse: ResponsesResponse = {
            id: responseId,
            object: "response",
            created: Math.floor(Date.now() / 1000),
            status: "completed",
            model: requestBody.model,
            output: [completedOutputItem],
          };
          const completedEvent: ResponsesStreamResponseCompleted = {
            type: "response.completed",
            response: completedResponse,
          };
          sendEvent("response.completed", completedEvent);
        } catch (error) {
          const errorEvent: ResponsesStreamError = {
            type: "error",
            code: "500",
            message: String(error),
            param: "",
          };
          sendEvent("error", errorEvent);
        }

        if (!res.writableEnded && !res.destroyed) {
          res.end();
        }
      } else {
        const result = await clonedAgent.run({ prompt: null });

        const messageContent: ResponsesMessageContent = {
          type: "output_text",
          text: result.result?.text ?? "",
        };

        const messageOutput: ResponsesMessageOutput = {
          type: "message",
          id: `msg_${uuidv4()}`,
          status: "completed",
          role: "assistant",
          content: [messageContent],
        };

        const response: ResponsesResponse = {
          id: responseId,
          object: "response",
          created: Math.floor(Date.now() / 1000),
          status: "completed",
          model: requestBody.model,
          output: [messageOutput],
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
          },
        };

        res.json(response);
      }
    } catch (error) {
      logger.error(error, "Error handling /responses request");
      if (!res.headersSent) {
        res.status(500).json({ error: String(error) });
      }
    }
  }
}
