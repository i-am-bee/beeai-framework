/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import express, { Request, Response, Router } from "express";
import { v4 as uuidv4 } from "uuid";
import { AnyAgent } from "@/agents/types.js";
import { transformRequestMessages } from "./utils.js";
import { ChatCompletionRequestBody, ChatCompletionResponse } from "./types.js";
import { Logger } from "@/logger/logger.js";

const logger = Logger.root.child({
  name: "OpenAI API",
});

export class ChatCompletionAPI {
  public readonly router: Router;

  constructor(
    private readonly modelFactory: (modelId: string) => Promise<AnyAgent>,
    private readonly apiKey?: string,
  ) {
    this.router = express.Router();
    this.router.use(express.json());

    this.router.post("/chat/completions", this.handler.bind(this));
  }

  private async handler(req: Request, res: Response) {
    const requestBody = req.body as ChatCompletionRequestBody;
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
      const messages = transformRequestMessages(requestBody.messages || []);
      const agent = await this.modelFactory(requestBody.model);

      // We clone the agent to avoid mutating the registered instance
      const clonedAgent = await agent.clone();
      clonedAgent.memory = await clonedAgent.memory.clone();
      clonedAgent.memory.reset();
      await clonedAgent.memory.addMany(messages);

      if (requestBody.stream) {
        const id = `chatcmpl-${uuidv4()}`;
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        // Abort the agent run if the client disconnects early
        const controller = new AbortController();
        req.on("close", () => {
          controller.abort();
        });

        try {
          await clonedAgent
            .run({ prompt: null }, { signal: controller.signal })
            .observe((emitter) => {
              // Match all events — required because the emitter is typed as
              // Emitter<unknown> at the AnyAgent abstraction. We emit text deltas
              // from "update" (ReActAgent incremental tokens) and from "success"
              // (ToolCallingAgent emits only start/success, with the final answer
              // in state.result). This keeps streaming working for both agents.
              emitter.match("*.*", async (eventData: any, event) => {
                let delta: string | undefined;
                if (event.name === "update") {
                  delta = eventData?.update?.value;
                } else if (event.name === "success") {
                  delta = eventData?.state?.result?.text;
                }
                if (!delta || res.writableEnded || res.destroyed) {
                  return;
                }
                const data = {
                  id,
                  object: "chat.completion.chunk",
                  model: requestBody.model,
                  created: Math.floor(Date.now() / 1000),
                  choices: [
                    {
                      index: 0,
                      delta: {
                        role: "assistant",
                        content: delta,
                      },
                      finish_reason: null,
                    },
                  ],
                };
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              });
            });

          if (!res.writableEnded && !res.destroyed) {
            const finalData = {
              id,
              object: "chat.completion.chunk",
              model: requestBody.model,
              created: Math.floor(Date.now() / 1000),
              choices: [
                {
                  index: 0,
                  delta: {},
                  finish_reason: "stop",
                },
              ],
            };
            res.write(`data: ${JSON.stringify(finalData)}\n\n`);
            res.write(`data: [DONE]\n\n`);
          }
        } catch (error) {
          logger.error(error, "Error during streaming agent run");
          if (!res.writableEnded && !res.destroyed) {
            // Send an error chunk so the client can detect failure
            const errData = {
              id,
              object: "chat.completion.chunk",
              model: requestBody.model,
              created: Math.floor(Date.now() / 1000),
              choices: [{ index: 0, delta: {}, finish_reason: "error" }],
              error: { message: String(error) },
            };
            try {
              res.write(`data: ${JSON.stringify(errData)}\n\n`);
              res.write(`data: [DONE]\n\n`);
            } catch (writeError) {
              logger.error(writeError, "Failed to write error response chunk");
            }
          }
        } finally {
          if (!res.writableEnded && !res.destroyed) {
            res.end();
          }
        }
      } else {
        const result = await clonedAgent.run({ prompt: null });

        const response: ChatCompletionResponse = {
          id: `chatcmpl-${uuidv4()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: requestBody.model,
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: result.result?.text ?? "",
              },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 0, // Mock usage since agents might not bubble it up uniformly yet
            completion_tokens: 0,
            total_tokens: 0,
          },
        };

        res.json(response);
      }
    } catch (error) {
      logger.error(error, "Error handling /chat/completions request");
      if (!res.headersSent) {
        res.status(500).json({ error: String(error) });
      }
    }
  }
}
