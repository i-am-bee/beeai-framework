/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { DefaultRunner } from "@/agents/react/runners/default/runner.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { AssistantMessage, Role, UserMessage } from "@/backend/message.js";
import { BaseMemory } from "@/memory/base.js";
import { ReActAgentUserPrompt } from "@/agents/react/prompts.js";
import { zip } from "remeda";
import { RunContext } from "@/context.js";
import { ReActAgent } from "@/agents/react/agent.js";

vi.mock("@/memory/tokenMemory.js", async () => {
  const { UnconstrainedMemory } = await import("@/memory/unconstrainedMemory.js");
  class TokenMemory extends UnconstrainedMemory {}
  return { TokenMemory };
});

vi.mock("@/context.js");

describe("ReAct Agent Runner", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  it("Handles different prompt input source", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2024-09-10T19:51:46.954Z"));

    const createMemory = async () => {
      const memory = new UnconstrainedMemory();
      await memory.addMany([
        new UserMessage("What is your name?"),
        new AssistantMessage("I am Bee"),
      ]);
      return memory;
    };

    const createInstance = async (memory: BaseMemory, prompt: string | null) => {
      const instance = new DefaultRunner(
        {
          llm: expect.any(Function),
          memory,
          tools: [],
          templates: {},
        },
        {},
        new RunContext<ReActAgent, any>({} as any, {} as any),
      );
      await instance.init({ prompt });
      return instance;
    };

    const memory = await createMemory();
    const prompt = "What can you do for me?";
    const instance = await createInstance(memory, prompt);

    const memory2 = await createMemory();
    await memory2.add(new UserMessage(prompt, { createdAt: new Date() }));
    const instance2 = await createInstance(memory2, null);
    expect(instance.memory.messages).toEqual(instance2.memory.messages);
  });

  it.each([
    ReActAgentUserPrompt.fork((old) => ({
      ...old,
      functions: { ...old.functions, formatMeta: () => "" },
    })),
    ReActAgentUserPrompt.fork((old) => ({ ...old, template: `{{input}}` })),
    ReActAgentUserPrompt.fork((old) => ({ ...old, template: `User: {{input}}` })),
    ReActAgentUserPrompt.fork((old) => ({ ...old, template: `` })),
  ])("Correctly formats user input", async (template: typeof ReActAgentUserPrompt) => {
    const memory = new UnconstrainedMemory();
    await memory.addMany([
      new UserMessage("What is your name?"),
      new AssistantMessage("Bee"),
      new UserMessage("Who are you?"),
      new AssistantMessage("I am a helpful assistant."),
    ]);

    const prompt = "What can you do for me?";
    const instance = new DefaultRunner(
      {
        llm: expect.any(Function),
        memory,
        tools: [],
        templates: {
          user: template,
        },
      },
      {},
      new RunContext<ReActAgent, any>({} as any, {} as any),
    );
    await instance.init({ prompt });

    for (const [a, b] of zip(
      [...memory.messages.filter((msg) => msg.role === Role.USER), new UserMessage(prompt)],
      instance.memory.messages.filter((msg) => msg.role === Role.USER),
    )) {
      expect(template.render({ input: a.text, meta: undefined })).toStrictEqual(b.text);
    }
  });
});
