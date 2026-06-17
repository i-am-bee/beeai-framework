/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import express, { Request, Response, Router } from "express";
import { v4 as uuidv4 } from "uuid";
import { AnyAgent } from "@/agents/types.js";
import { Logger } from "@/logger/logger.js";
import { ReActAgentRunOutput } from "@/agents/react/types.js";
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

    const authHeader = req.headers.authorization;
    if (this.apiKey) {
      if (!authHeader || authHeader.replace("Bearer ", "") !== this.apiKey) {
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

        const sendEvent = (eventType: string, data: unknown) => {
          sequenceNumber++;
          res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
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
          sequence_number: sequenceNumber,
        };
        sendEvent("response.created", createdEvent);

        // response.in_progress
        const inProgressEvent: ResponsesStreamResponseInProgress = {
          type: "response.in_progress",
          response: { ...createdResponse },
          sequence_number: sequenceNumber,
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
          sequence_number: sequenceNumber,
        };
        sendEvent("response.output_item.added", outputItemAddedEvent);

        // response.content_part.added
        const emptyPart: ResponsesStreamPartOutputText = {
          text: "",
          type: "response.output_text.done",
          annotations: [],
        };
        const contentPartAddedEvent: ResponsesStreamContentPartAdded = {
          type: "response.content_part.added",
          content_index: 0,
          item_id: outputItemId,
          output_index: 0,
          part: emptyPart,
          sequence_number: sequenceNumber,
        };
        sendEvent("response.content_part.added", contentPartAddedEvent);

        // Listen to agent emitter 'update' events for streaming deltas
        (clonedAgent.emitter as any).on("update", async ({ update }: any) => {
          const delta = update.value;
          accumulatedText += delta;

          const deltaEvent: ResponsesStreamOutputTextDelta = {
            type: "response.output_text.delta",
            content_index: 0,
            delta,
            item_id: outputItemId,
            output_index: 0,
            sequence_number: sequenceNumber,
          };
          sendEvent("response.output_text.delta", deltaEvent);
        });

        try {
          await clonedAgent.run({ prompt: null });

          // response.output_text.done
          const textDoneEvent: ResponsesStreamOutputTextDone = {
            type: "response.output_text.done",
            content_index: 0,
            text: accumulatedText,
            item_id: outputItemId,
            output_index: 0,
            sequence_number: sequenceNumber,
          };
          sendEvent("response.output_text.done", textDoneEvent);

          // response.content_part.done
          const finalPart: ResponsesStreamPartOutputText = {
            text: accumulatedText,
            type: "response.output_text.done",
            annotations: [],
          };
          const contentPartDoneEvent: ResponsesStreamContentPartDone = {
            type: "response.content_part.done",
            content_index: 0,
            item_id: outputItemId,
            output_index: 0,
            part: finalPart,
            sequence_number: sequenceNumber,
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
            sequence_number: sequenceNumber,
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
            sequence_number: sequenceNumber,
          };
          sendEvent("response.completed", completedEvent);
        } catch (error) {
          const errorEvent: ResponsesStreamError = {
            type: "error",
            code: "500",
            message: String(error),
            param: "",
            sequence_number: sequenceNumber,
          };
          sendEvent("error", errorEvent);
        }

        res.end();
      } else {
        const result = (await clonedAgent.run({ prompt: null })) as ReActAgentRunOutput;

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
      res.status(500).json({ error: String(error) });
    }
  }
}
