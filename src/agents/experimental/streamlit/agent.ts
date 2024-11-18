/**
 * Copyright 2024 IBM Corp.
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

import { BaseAgent, BaseAgentRunOptions } from "@/agents/base.js";
import { AgentMeta } from "@/agents/types.js";
import { GetRunContext } from "@/context.js";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { BaseMemory } from "@/memory/base.js";
import { ChatLLM, ChatLLMOutput } from "@/llms/chat.js";
import { isTruthy } from "remeda";
import { BaseMessage, Role } from "@/llms/primitives/message.js";
import { TokenMemory } from "@/memory/tokenMemory.js";
import {
  StreamlitAgentSystemPrompt,
  StreamlitAgentTemplates,
} from "@/agents/experimental/streamlit/prompts.js";
import { findFirstPair } from "@/internals/helpers/string.js";
import { BaseLLMOutput } from "@/llms/base.js";

interface Input {
  llm: ChatLLM<ChatLLMOutput>;
  memory: BaseMemory;
  templates?: Partial<StreamlitAgentTemplates>;
}

interface RunInput {
  prompt: string | null;
}

interface Options extends BaseAgentRunOptions {}

interface Result {
  raw: string;

  app: string | null;
  text: string;
}

interface RunOutput {
  result: Result;
  message: BaseMessage;
  memory: BaseMemory;
}

interface Events {
  newToken: Callback<{
    delta: string;
    state: Readonly<{
      content: string;
    }>;
    chunk: BaseLLMOutput;
  }>;
}

export class StreamlitAgent extends BaseAgent<RunInput, RunOutput, Options> {
  public emitter = new Emitter<Events>({
    namespace: ["agent", "experimental", "streamlit"],
    creator: this,
  });

  constructor(protected readonly input: Input) {
    super();
  }

  protected async _run(
    input: RunInput,
    _options: Options | undefined,
    run: GetRunContext<typeof this>,
  ): Promise<RunOutput> {
    const systemMessage = BaseMessage.of({
      role: Role.SYSTEM,
      text: (this.input.templates?.system ?? StreamlitAgentSystemPrompt).render({}),
    });

    const userMessage =
      input.prompt &&
      BaseMessage.of({
        role: Role.USER,
        text: input.prompt,
        meta: {
          createdAt: new Date(),
        },
      });

    const runMemory = new TokenMemory({ llm: this.input.llm });
    await runMemory.addMany(
      [systemMessage, ...this.input.memory.messages, userMessage].filter(isTruthy),
    );

    let content = "";
    for await (const chunk of this.input.llm.stream(runMemory.messages, { signal: run.signal })) {
      const delta = chunk.getTextContent();
      content += delta;
      await run.emitter.emit("newToken", { delta, state: { content }, chunk });
    }

    const assistantMessage = BaseMessage.of({
      role: Role.ASSISTANT,
      text: content,
    });
    await this.memory.addMany([userMessage, assistantMessage].filter(isTruthy));

    const match = findFirstPair(content, ["```python-app\n", "```\n"]);
    return {
      result: {
        raw: content,
        app: match ? match.inner : null,
        text: match
          ? `${content.substring(0, match.start)}${content.substring(match.end + 1)}`
          : content,
      },
      message: assistantMessage,
      memory: runMemory,
    };
  }

  public get meta(): AgentMeta {
    return {
      name: `Streamlit`,
      tools: [],
      description: `StreamlitAgent is an experimental meta-app agent that uses \`Meta LLaMa 3.1 70B\` to build \`IBM Granite 3 8B\`-powered apps using Streamlit -- a simple UI framework for Python.`,
    };
  }

  public get memory() {
    return this.input.memory;
  }

  createSnapshot() {
    return {
      input: this.input,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
