/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { RequirementAgent } from "./agent.js";
import { RequirementAgentRunner } from "./runner.js";
import { RequirementsReasoner } from "./utils/llm.js";
import { ChatModel, ChatModelInput, ChatModelOutput } from "@/backend/chat.js";
import { AssistantMessage, ToolMessage } from "@/backend/message.js";
import { GetRunContext } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { Tool, ToolEvents, StringToolOutput } from "@/tools/base.js";
import { OpenAPITool } from "@/tools/openapi.js";
import { vi } from "vitest";
import { z } from "zod";

class BusinessTool extends Tool {
  name = "business";
  description = "A caller-owned tool";
  emitter = Emitter.root.child<ToolEvents>({ namespace: ["tool", "business"], creator: this });
  inputSchema() {
    return z.object({});
  }
  async _run() {
    return new StringToolOutput("business result");
  }
}
class ScriptedModel extends ChatModel {
  modelId = "scripted";
  providerId = "test";
  emitter = Emitter.root.child({ namespace: ["model", "scripted"], creator: this });
  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
  mode: "text" | "tool" | "invalid" | "error" | "cancel" = "tool";
  started: () => void = () => {};
  async _create(_input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    this.started();
    if (this.mode === "error") {
      throw new Error("Scripted model failure");
    }
    if (this.mode === "cancel") {
      await new Promise<void>((_resolve, reject) => {
        ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
      });
    }
    if (this.mode === "invalid") {
      this.mode = "tool";
      return new ChatModelOutput([new AssistantMessage("")]);
    }
    return new ChatModelOutput([
      new AssistantMessage(
        this.mode === "text"
          ? '{"response":"hello"}'
          : [
              {
                type: "tool-call",
                toolName: "final_answer",
                toolCallId: "answer",
                input: { response: "hello" },
              },
            ],
      ),
    ]);
  }
  async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
    yield await this._create(input, ctx);
  }
}
const rootChildren = () => Emitter.root.createSnapshot().cleanups.length;
function fixture(saveIntermediateSteps = true) {
  const model = new ScriptedModel();
  const tool = new BusinessTool();
  const agent = new RequirementAgent({
    llm: model,
    saveIntermediateSteps,
    tools: [tool],
    finalAnswerAsTool: false,
    execution: { maxRetriesPerStep: 0, totalMaxRetries: 0 },
  });
  const baseline = rootChildren();
  const owned: Emitter<any>[] = [];
  const child = Emitter.root.child.bind(Emitter.root);
  vi.spyOn(Emitter.root, "child").mockImplementation((input) => {
    const emitter = child(input);
    if (
      ["middleware.streamToolCall", "tool.final_answer"].includes(input?.namespace?.join(".") ?? "")
    ) {
      owned.push(emitter);
    }
    return emitter;
  });
  return {
    model,
    tool,
    agent,
    baseline,
    cleanup() {
      vi.restoreAllMocks();
      owned.forEach((emitter) => emitter.destroy());
      agent.destroy();
      model.destroy();
      tool.emitter.destroy();
    },
  };
}
async function healthyRun(agent: RequirementAgent) {
  const output = await agent.run({ prompt: "hello" });
  expect(output.result.text).toBe("hello");
  expect(output.state.result).toEqual({ response: "hello" });
  expect(output.memory.messages.at(-1)?.role).toBe("tool");
  expect(agent.memory.messages).toEqual(output.memory.messages);
}
describe("RequirementAgent owned emitter lifecycle", () => {
  it.each(["text", "tool"] as const)(
    "releases repeated %s answers without removing business subscriptions",
    async (mode) => {
      const f = fixture();
      const finalAnswer = vi.fn();
      const rootAnswer = vi.fn();
      const finalToolSuccess = vi.fn();
      const unmatchTool = Emitter.root.match(
        (meta) => meta.path === "tool.final_answer.success",
        finalToolSuccess,
      );
      const modelSuccess = vi.fn();
      const toolSuccess = vi.fn();
      const unmatch = Emitter.root.match(
        (meta) => meta.path === "agent.requirement.finalAnswer",
        rootAnswer,
      );
      f.agent.emitter.on("finalAnswer", finalAnswer);
      f.model.emitter.on("success", modelSuccess);
      f.tool.emitter.on("success", toolSuccess);
      f.model.mode = mode;
      try {
        for (let i = 0; i < 3; i++) {
          await healthyRun(f.agent);
          expect(rootChildren()).toBe(f.baseline);
          f.agent.memory.reset();
        }
        expect(finalAnswer).toHaveBeenCalledTimes(3);
        expect(rootAnswer).toHaveBeenCalledTimes(3);
        expect(finalToolSuccess).toHaveBeenCalledTimes(3);
        expect(modelSuccess).toHaveBeenCalledTimes(3);
        await expect(f.tool.run({})).resolves.toEqual(new StringToolOutput("business result"));
        expect(toolSuccess).toHaveBeenCalledTimes(1);
        expect(rootChildren()).toBe(f.baseline);
      } finally {
        unmatch();
        unmatchTool();
        f.cleanup();
      }
    },
  );
  it("releases model failure and permits another run", async () => {
    const f = fixture();
    f.model.mode = "error";
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      f.model.mode = "tool";
      await healthyRun(f.agent);
      expect(rootChildren()).toBe(f.baseline);
    } finally {
      f.cleanup();
    }
  });
  it("releases in-flight cancellation and permits another run", async () => {
    const f = fixture();
    const controller = new AbortController();
    let started!: () => void;
    const ready = new Promise<void>((resolve) => {
      started = resolve;
    });
    f.model.started = started;
    f.model.mode = "cancel";
    const running = Promise.resolve(
      f.agent.run({ prompt: "hello" }, { signal: controller.signal }),
    );
    const result = expect(running).rejects.toThrow();
    try {
      await ready;
      controller.abort(new Error("Scripted cancellation"));
      await result;
      expect(controller.signal.aborted).toBe(true);
      expect(rootChildren()).toBe(f.baseline);
      f.model.mode = "tool";
      await healthyRun(f.agent);
      expect(rootChildren()).toBe(f.baseline);
    } finally {
      f.cleanup();
    }
  });
  it("releases each middleware during invalid-answer retry", async () => {
    const f = fixture();
    f.model.mode = "invalid";
    try {
      const output = await f.agent.run(
        { prompt: "hello" },
        { execution: { maxRetriesPerStep: 1, totalMaxRetries: 1 } },
      );
      expect(output.result.text).toBe("hello");
      expect(output.state.result).toEqual({ response: "hello" });
      expect(rootChildren()).toBe(f.baseline);
    } finally {
      f.cleanup();
    }
  });
  it.each(["text", "tool"] as const)("releases %s final-answer listener failures", async (mode) => {
    const f = fixture();
    f.model.mode = mode;
    const unsubscribe = f.agent.emitter.on(
      "finalAnswer",
      () => {
        throw new Error("Listener failure");
      },
      { isBlocking: true },
    );
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      unsubscribe();
      await healthyRun(f.agent);
      expect(rootChildren()).toBe(f.baseline);
    } finally {
      unsubscribe();
      f.cleanup();
    }
  });
  it("releases middleware when request preparation fails", async () => {
    const f = fixture();
    vi.spyOn(RequirementAgentRunner.prototype as any, "prepareLLMRequest").mockRejectedValueOnce(
      new Error("Request failure"),
    );
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      await healthyRun(f.agent);
    } finally {
      f.cleanup();
    }
  });
  it("releases the final-answer emitter when answer validation fails", async () => {
    const f = fixture();
    const run = RequirementAgentRunner.prototype.run;
    vi.spyOn(RequirementAgentRunner.prototype, "run").mockImplementationOnce(async function (
      this: RequirementAgentRunner,
    ) {
      const state = await run.call(this);
      state.answer = null;
      return state;
    });
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      await healthyRun(f.agent);
    } finally {
      f.cleanup();
    }
  });
  it("releases input-processing failures before runner.run", async () => {
    const f = fixture();
    vi.spyOn(f.agent as any, "processInput").mockImplementationOnce(() => {
      throw new Error("Input failure");
    });
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      await healthyRun(f.agent);
    } finally {
      f.cleanup();
    }
  });
  it("releases requirement initialization failures", async () => {
    const f = fixture();
    vi.spyOn(RequirementsReasoner.prototype, "update").mockRejectedValueOnce(
      new Error("Requirement failure"),
    );
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      await healthyRun(f.agent);
    } finally {
      f.cleanup();
    }
  });

  it("releases initial memory failures before runner.run", async () => {
    const f = fixture();
    vi.spyOn(RequirementAgentRunner.prototype, "addMessages").mockRejectedValueOnce(
      new Error("Initial memory failure"),
    );
    try {
      await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
      expect(rootChildren()).toBe(f.baseline);
      await healthyRun(f.agent);
    } finally {
      f.cleanup();
    }
  });
  it.each([true, false])(
    "releases final memory persistence failures with intermediate steps=%s",
    async (saveIntermediateSteps) => {
      const f = fixture(saveIntermediateSteps);
      vi.spyOn(f.agent.memory, "addMany").mockRejectedValueOnce(new Error("Memory failure"));
      try {
        await expect(f.agent.run({ prompt: "hello" })).rejects.toThrow();
        expect(rootChildren()).toBe(f.baseline);
        await healthyRun(f.agent);
      } finally {
        f.cleanup();
      }
    },
  );
});
describe("RequirementAgent maxIterations", () => {
  it.each([
    [0, false],
    [1, true],
    [undefined, true],
  ])("with maxIterations=%s resolves=%s", async (maxIterations, resolves) => {
    const f = fixture();
    try {
      const run = f.agent.run({ prompt: "hello" }, { execution: { maxIterations } });
      if (resolves) {
        await expect(run).resolves.toBeDefined();
      } else {
        await expect(run).rejects.toThrow("not able to resolve the task");
      }
    } finally {
      f.cleanup();
    }
  });
});

describe("RequirementAgent OpenAPI responses", () => {
  it.each([
    { status: 401, expected: "HTTP 401" },
    { status: 204, expected: "No results were found!" },
  ])("passes the outcome of HTTP $status to the model", async ({ status, expected }) => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal("fetch", fetchMock);
    const tool = new OpenAPITool({
      openApiSchema: JSON.stringify({
        openapi: "3.0.0",
        info: { title: "test_api", version: "1.0.0" },
        servers: [{ url: "https://example.com" }],
        paths: { "/items": { get: { responses: { "204": { description: "No content" } } } } },
      }),
    });
    const model = new ScriptedModel();
    const create = vi.spyOn(model, "_create").mockResolvedValueOnce(
      new ChatModelOutput([
        new AssistantMessage([
          {
            type: "tool-call",
            toolCallId: "read-items",
            toolName: tool.name,
            input: { path: "/items", method: "get" },
          },
        ]),
      ]),
    );
    const agent = new RequirementAgent({ llm: model, tools: [tool] });

    try {
      const result = await agent.run({ prompt: "Read the items." });
      expect(result.result.text).toBe("hello");
      expect(fetchMock).toHaveBeenCalledOnce();
      const messages = create.mock.calls[1]?.[0].messages ?? [];
      const toolResult = messages
        .filter((message): message is ToolMessage => message instanceof ToolMessage)
        .flatMap((message) => message.getToolResults())
        .find((part) => part.toolCallId === "read-items");
      expect(toolResult?.output).toEqual({
        type: "text",
        value: expect.stringContaining(expected),
      });
    } finally {
      agent.destroy();
      model.destroy();
      tool.emitter.destroy();
      vi.unstubAllGlobals();
    }
  });
});

describe("RequirementAgent plain-text final answers", () => {
  class TextReplyModel extends ChatModel {
    modelId = "text-reply";
    providerId = "test";
    emitter = Emitter.root.child({ namespace: ["model", "textReply"], creator: this });
    constructor(public readonly reply: string) {
      super();
    }
    loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
      Object.assign(this, snapshot);
    }
    async _create(_input: ChatModelInput, _ctx: GetRunContext<typeof this>) {
      return new ChatModelOutput([new AssistantMessage(this.reply)]);
    }
    async *_createStream(input: ChatModelInput, ctx: GetRunContext<typeof this>) {
      yield await this._create(input, ctx);
    }
  }

  async function runWithReply(
    reply: string,
    options: { finalAnswerAsTool: boolean; expectedOutput?: z.ZodSchema },
  ) {
    const model = new TextReplyModel(reply);
    const tool = new BusinessTool();
    const agent = new RequirementAgent({
      llm: model,
      tools: [tool],
      finalAnswerAsTool: options.finalAnswerAsTool,
      execution: { maxRetriesPerStep: 0, totalMaxRetries: 0 },
    });
    try {
      return await agent.run({ prompt: "hello", expectedOutput: options.expectedOutput });
    } finally {
      agent.destroy();
      model.destroy();
      tool.emitter.destroy();
    }
  }

  it.each([
    ["plain text", "Hello world"],
    ["a number", "42"],
    ["a JSON array", "[1, 2]"],
    ["a JSON object fragment", 'Send `{"name": "x"}` as the request body.'],
    ["an empty object", "Use {} for no options."],
  ])("keeps a reply with %s as the whole response", async (_, reply) => {
    for (const finalAnswerAsTool of [false, true]) {
      const output = await runWithReply(reply, { finalAnswerAsTool });
      expect(output.result.text).toBe(reply);
      expect(output.state.result).toEqual({ response: reply });
    }
  });

  it("uses a reply that is a valid final answer object", async () => {
    const output = await runWithReply('{"response": "hello"}', { finalAnswerAsTool: false });
    expect(output.result.text).toBe("hello");
    expect(output.state.result).toEqual({ response: "hello" });
  });

  it("passes a JSON object to a custom expected output schema", async () => {
    const output = await runWithReply('Result: {"city": "Prague", "temperature": 21}', {
      finalAnswerAsTool: false,
      expectedOutput: z.object({ city: z.string(), temperature: z.number() }),
    });
    expect(output.state.result).toEqual({ city: "Prague", temperature: 21 });
  });
});
