/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { FrameworkError } from "@/errors.js";
import { beforeEach, expect, vi } from "vitest";
import { Logger } from "@/logger/logger.js";
import { ReActAgent } from "@/agents/react/agent.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { Message } from "@/backend/message.js";
import { createCallbackRegister } from "@tests/e2e/utils.js";
import { omitEmptyValues } from "@/internals/helpers/object.js";
import * as process from "node:process";
import { createChatLLM } from "@tests/utils/llmFactory.js";
import { ReActAgentMeta } from "@/agents/react/types.js";
import { GoogleSearchTool } from "@/tools/search/googleSearch.js";

const googleSearchApiKey = process.env.GOOGLE_API_KEY;
const googleSearchCseId = process.env.GOOGLE_CSE_ID;

describe.runIf(Boolean(googleSearchApiKey && googleSearchCseId))("ReAct Agent", () => {
  const createAgent = () => {
    return new ReActAgent({
      llm: createChatLLM(),
      memory: new UnconstrainedMemory(),
      tools: [
        new GoogleSearchTool({
          apiKey: googleSearchApiKey,
          cseId: googleSearchCseId,
          maxResults: 10,
        }),
      ],
    });
  };

  beforeEach(() => {
    vi.useRealTimers();
  });

  it("Aborts", async () => {
    const abortController = new AbortController();
    const myError = new Error("Stop!");
    setTimeout(() => {
      abortController.abort(myError);
    }, 500);

    const agent = createAgent();
    try {
      await agent.run(
        {
          prompt: 'What"s the biggest building on the world?"',
        },
        {
          signal: abortController.signal,
          execution: {
            maxIterations: 10,
            totalMaxRetries: 0,
            maxRetriesPerStep: 0,
          },
        },
      );
    } catch (e) {
      expect(e).toBeInstanceOf(FrameworkError);
      expect(e.getCause()).toBe(myError);
    }
  });

  it("Runs", async () => {
    const callbacks = createCallbackRegister();
    const userLogger = Logger.of({ name: "user" });
    const agent = createAgent();

    let response: Awaited<ReturnType<typeof agent.run>>;
    try {
      response = await agent
        .run(
          { prompt: "Who is the president of Czech Republic?" },
          {
            execution: {
              maxIterations: 5,
              totalMaxRetries: 5,
            },
          },
        )
        .observe((emitter) => {
          let lastIteration = 0;
          emitter.match("*", (data: { meta: ReActAgentMeta }, event) => {
            expect(data.meta.iteration >= lastIteration);
            expect(event.groupId).toBeDefined();
            lastIteration = data.meta.iteration;
          });
          emitter.registerCallbacks({
            success: callbacks.create("success", {
              check: ({ data }) => {
                expect(data).toBeInstanceOf(Message);
                expect(Object.keys(omitEmptyValues(data)).length).toBeGreaterThan(0);
              },
            }),
            retry: callbacks.create("retry", { required: false }),
            error: callbacks.create("error", { required: false }),
            start: callbacks.create("start"),
            update: callbacks.create("update"),
            toolStart: callbacks.create("toolStart"),
            toolSuccess: callbacks.create("toolSuccess"),
            toolError: callbacks.create("toolError", {
              required: false,
              check: ({ data }) => {
                expect(data.error).toBeInstanceOf(FrameworkError);
                // eslint-disable-next-line no-console
                console.warn("Tool Error", data.error.explain());
              },
            }),
          });
        });
    } catch (e) {
      expect(e).toBeInstanceOf(FrameworkError);
      throw e;
    } finally {
      agent.destroy();
    }

    userLogger.info("==================END=====================");
    expect(response.result).toBeDefined();
    expect(response.result.text).toMatch(/petr pavel/i);

    callbacks.verify(({ fn }) => {
      for (const [_, event, ...extra] of fn.mock.calls) {
        expect(event).toBeDefined();
        expect(extra).toHaveLength(0);
      }
    });
  });
});
