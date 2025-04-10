/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import { AgentError, BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { AgentMeta } from "@/agents/types.js";
import { GetRunContext } from "@/context.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { BaseMemory } from "@/memory/base.js";
import { isTruthy, last } from "remeda";
import { AssistantMessage, Message, Role, SystemMessage, UserMessage } from "@/backend/message.js";
import {
  StreamlitAgentSystemPrompt,
  StreamlitAgentTemplates,
} from "@/agents/experimental/streamlit/prompts.js";
import { TokenMemory } from "@/memory/tokenMemory.js";
import { findFirstPair } from "@/internals/helpers/string.js";
import { ChatModel, ChatModelOutput } from "@/backend/chat.js";

export interface StreamlitAgentInput {
  llm: ChatModel;
  memory: BaseMemory;
  templates?: Partial<StreamlitAgentTemplates>;
}

export interface StreamlitRunInput {
  prompt: string | null;
}

interface Block {
  name: "text" | "app";
  content: string;
  start: number;
  end: number;
}

interface Result {
  raw: string;
  blocks: Block[];
}

export interface StreamlitRunOutput {
  result: Result;
  message: Message;
  memory: BaseMemory;
}

export interface StreamlitEvents {
  newToken: Callback<{
    delta: string;
    state: Readonly<{
      content: string;
    }>;
    chunk: ChatModelOutput;
  }>;
}

export class StreamlitAgent extends BaseAgent<StreamlitRunInput, StreamlitRunOutput> {
  public emitter = new Emitter<StreamlitEvents>({
    namespace: ["agent", "experimental", "streamlit"],
    creator: this,
  });

  constructor(protected readonly input: StreamlitAgentInput) {
    super();
  }

  public get meta(): AgentMeta {
    return {
      name: `Streamlit`,
      tools: [],
      description: `StreamlitAgent is an experimental meta-app agent that uses \`Meta LLaMa 3.1 70B\` to build \`IBM Granite 3 8B\`-powered apps using Streamlit -- a simple UI framework for Python.`,
    };
  }

  set memory(memory: BaseMemory) {
    this.input.memory = memory;
  }

  public get memory() {
    return this.input.memory;
  }

  protected async _run(
    input: StreamlitRunInput,
    _options: BaseAgentRunOptions,
    run: GetRunContext<typeof this>,
  ): Promise<StreamlitRunOutput> {
    const { userMessage, runMemory } = await this.prepare(input);

    let content = "";
    const raw = await this.input.llm
      .create({
        stream: true,
        messages: runMemory.messages,
        abortSignal: run.signal,
      })
      .observe((emitter) => {
        emitter.on("newToken", async ({ value: chunk }) => {
          const delta = chunk.getTextContent();
          if (delta) {
            content += delta;
            await run.emitter.emit("newToken", { delta, state: { content }, chunk });
          }
        });
      });
    const result = this.parse(content || raw.getTextContent());

    const assistantMessage = new AssistantMessage(content);
    await this.memory.addMany([userMessage, assistantMessage].filter(isTruthy));

    return {
      result,
      message: assistantMessage,
      memory: runMemory,
    };
  }

  protected async prepare(input: StreamlitRunInput) {
    const systemMessage = new SystemMessage(
      (this.input.templates?.system ?? StreamlitAgentSystemPrompt).render({}),
    );

    const userMessage =
      input.prompt !== null || this.memory.isEmpty()
        ? new UserMessage(input.prompt ?? "No message.", { createdAt: new Date() })
        : null;

    const inputMessages = [...this.memory.messages, userMessage].filter(isTruthy);

    const runMemory = new TokenMemory({
      capacityThreshold: 0.85,
      syncThreshold: 0.6,
      handlers: {
        removalSelector(msgs) {
          // First we remove messages from the past conversations
          const prevConversationMsg = msgs.find((msg) => inputMessages.includes(msg));
          if (prevConversationMsg && prevConversationMsg !== last(inputMessages)) {
            return prevConversationMsg;
          }

          const lastMsg = msgs.length > 3 && msgs.find((m) => m.role === Role.ASSISTANT);
          if (!lastMsg) {
            throw new AgentError("Cannot fit the current conversation into the context window!");
          }
          return lastMsg;
        },
      },
    });
    await runMemory.addMany([systemMessage, ...inputMessages].filter(isTruthy));

    return { runMemory, userMessage };
  }

  protected parse(raw: string): Result {
    const blocks: Block[] = [];

    for (let i = 0; i < raw.length; ) {
      const text = raw.substring(i);

      const code = findFirstPair(text, ["```python-app\n", "\n```"], { allowOverlap: true });
      if (!code) {
        blocks.push({ start: i, end: i + text.length, content: text, name: "text" });
        break;
      }

      if (code.start > 0) {
        blocks.push({
          name: "text",
          start: i,
          end: i + code.start,
          content: text.substring(0, code.start),
        });
      }
      blocks.push({
        name: "app",
        content: code.inner,
        start: i + code.start,
        end: i + code.end,
      });
      i += code.end;
    }

    return {
      raw,
      blocks,
    };
  }

  createSnapshot() {
    return {
      ...super.createSnapshot(),
      input: this.input,
    };
  }
}
