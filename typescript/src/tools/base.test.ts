/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseToolOptions,
  BaseToolRunOptions,
  ToolEmitter,
  DynamicTool,
  JSONToolOutput,
  StringToolOutput,
  Tool,
  ToolError,
  ToolInput,
} from "@/tools/base.js";
import { AnyFn } from "@/internals/types.js";
import { setTimeout } from "node:timers/promises";
import { beforeEach, expect, vi } from "vitest";
import { z } from "zod";

import { SlidingCache } from "@/cache/slidingCache.js";
import { Serializer } from "@/serializer/serializer.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";
import { Emitter } from "@/emitter/emitter.js";

describe("JSON tool output", () => {
  test.each([0, -0, false, true, 1, -2, 0.5])(
    "preserves scalar %j through serialization",
    async (value) => {
      const output = new JSONToolOutput(value);
      const restored = await JSONToolOutput.fromSerialized<JSONToolOutput<typeof value>>(
        await output.serialize(),
      );
      expect(restored.result).toBe(value === 0 ? 0 : value);
      expect(restored.isEmpty()).toBe(false);
      expect(restored.getTextContent()).toBe(JSON.stringify(value));
    },
  );

  test("keeps serialization behavior for non-JSON scalars", () => {
    expect(new JSONToolOutput(undefined).getTextContent()).toBeUndefined();
    for (const value of [NaN, Infinity, -Infinity]) {
      const output = new JSONToolOutput(value);
      expect(output.isEmpty()).toBe(true);
      expect(output.getTextContent()).toBe("null");
    }
    for (const value of [0n, 1n]) {
      const output = new JSONToolOutput(value);
      expect(output.isEmpty()).toBe(true);
      expect(() => output.getTextContent()).toThrow(TypeError);
    }
  });
});

describe("Base Tool", () => {
  beforeEach(() => {
    vi.clearAllTimers();
  });

  class FatalError extends Error {}

  const createDummyTool = <T extends AnyFn>(options: BaseToolOptions, fn: T) => {
    class DummyTool extends Tool<StringToolOutput> {
      name = "DummyTool";
      description = "DummyTool description";
      emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
        namespace: ["tool", "dummy"],
        creator: this,
      });

      inputSchema() {
        return z.object({ query: z.string() });
      }

      protected async _run(
        { query }: ToolInput<this>,
        options: Partial<BaseToolRunOptions>,
      ): Promise<StringToolOutput> {
        const result = await fn(query, options);
        return new StringToolOutput(result);
      }
    }

    const instance = new DummyTool(options);
    Serializer.deregister(DummyTool);
    return vi.mocked(instance) as typeof instance;
  };

  interface SettingsTestCase {
    query: string;
    output: string;
    runOptions?: BaseToolRunOptions;
  }
  it.each([
    {
      query: "Hello!",
      output: "World!",
      runOptions: undefined,
    },
  ] as SettingsTestCase[])(
    "Implementation is correctly called (%o)",
    async ({ query, output, runOptions }) => {
      const handler = vi.fn();
      const tool = createDummyTool({}, handler);

      handler.mockResolvedValue(output);
      await expect(tool.run({ query }, runOptions)).resolves.toBeTruthy();
      expect(handler).toBeCalledTimes(1);
      expect(handler).toBeCalledWith(query, runOptions ?? {});
    },
  );

  interface Retries {
    options: {
      fatalErrors: BaseToolOptions["fatalErrors"];
      retryOptions: BaseToolOptions["retryOptions"];
    };
    errors?: Error[];
    expected: {
      totalCalls: number;
    };
  }
  it.each([
    {
      options: {
        fatalErrors: [],
        retryOptions: {
          maxRetries: 5,
          factor: 0,
        },
      },
      expected: {
        totalCalls: 6,
      },
    },
    {
      options: {
        fatalErrors: [FatalError],
        retryOptions: {
          maxRetries: 100,
          factor: 0,
        },
      },
      errors: [new Error("Test"), new FatalError("Fatal!")],
      expected: {
        totalCalls: 2,
      },
    },
    {
      options: {
        fatalErrors: [],
        retryOptions: {
          maxRetries: 0,
          factor: 0,
        },
      },
      expected: {
        totalCalls: 1,
      },
    },
    {
      options: {
        fatalErrors: [Error],
        retryOptions: {
          maxRetries: 5,
          factor: 0,
        },
      },
      errors: [new Error("Test")],
      expected: {
        totalCalls: 1,
      },
    },
  ] as Retries[])("Correctly handles retries (%i)", async ({ errors, options, expected }) => {
    const handler = vi.fn(
      (() => {
        if (errors) {
          return () => {
            const err = errors.shift();
            expect(err).toBeDefined();
            throw err;
          };
        } else {
          return () => {
            throw new Error("Error!");
          };
        }
      })(),
    );

    const tool = createDummyTool(options, handler);
    await expect(tool.run({ query: "Hello!" })).rejects.toThrowError(
      expect.objectContaining({
        name: ToolError.name,
        message: 'Tool "DummyTool" has occurred an error!',
      }),
    );
    expect(handler).toBeCalledTimes(expected.totalCalls);
  });

  it("Applies the default backoff factor when only maxRetries is set", async () => {
    vi.useRealTimers();

    const handler = vi.fn(() => {
      throw new Error("Error!");
    });
    const tool = createDummyTool({ retryOptions: { maxRetries: 1 } }, handler);

    const startedAt = Date.now();
    await expect(tool.run({ query: "Hello!" })).rejects.toThrowError(ToolError);
    const elapsed = Date.now() - startedAt;

    expect(handler).toBeCalledTimes(2);
    // The default factor is 1, so the single retry waits 1s. A factor of 0 retries instantly.
    expect(elapsed).toBeGreaterThanOrEqual(900);
  });

  it("Aborts on signal", async () => {
    vi.useRealTimers();

    const handler = vi.fn();
    handler.mockImplementationOnce(async () => {
      await setTimeout(50);
      throw new Error("First error!");
    });
    handler.mockImplementationOnce(async () => {
      await setTimeout(200);
      expect(true).toBe(false); // should never be called!
    });

    const tool = createDummyTool(
      {
        retryOptions: {
          maxRetries: 5,
          factor: 0,
        },
      },
      handler,
    );

    const controller = new AbortController();
    const abortError = new Error("Action has been cancelled!");
    void setTimeout(150).then(() => controller.abort(abortError));

    try {
      const task = tool.run({ query: "Hello!" }, { signal: controller.signal });
      await task;
      expect(true).toBe(false);
    } catch (e) {
      expect(e).toBeInstanceOf(ToolError);
      expect((e as ToolError).errors[0]).toBe(abortError);
    }
    expect(handler).toBeCalledTimes(2);
  });

  interface CacheTestInput {
    options: BaseToolOptions;
    inputs: {
      query: string;
      options: BaseToolOptions;
      cached: boolean;
      sleep?: number;
    }[];
  }
  it.each([
    {
      options: { cache: new SlidingCache({ size: 10 }) },
      inputs: [
        { query: "IBM Research", options: {}, cached: false },
        { query: "IBM Research", options: {}, cached: true },
        { query: "IBM Research", options: {}, cached: true },
        { query: "IBM Software", options: {}, cached: false },
        {
          query: "Who was Alan Turing?",
          options: {},
          cached: false,
        },
      ],
    },
    {
      options: { cache: new SlidingCache({ size: 2 }) },
      inputs: [
        { query: "A", options: {}, cached: false },
        { query: "B", options: {}, cached: false },
        { query: "A", options: {}, cached: true },
        { query: "B", options: {}, cached: true },
        { query: "C", options: {}, cached: false },
        { query: "B", options: {}, cached: true },
        { query: "C", options: {}, cached: true },
        { query: "A", options: {}, cached: false },
      ],
    },
    {
      options: { cache: new SlidingCache({ size: 10 }) },
      inputs: [
        { query: "A", cached: false },
        { query: "A", cached: true },
        null, // clear cache
        { query: "A", cached: false },
        { query: "A", cached: true },
      ],
    },
    {
      options: { cache: new SlidingCache({ size: 10, ttl: 1000 }) },
      inputs: [
        { query: "A", cached: false },
        { query: "B", cached: false },
        { query: "A", cached: true },
        { query: "B", cached: true },
        { sleep: 1500 },
        { query: "A", cached: false },
        { query: "B", cached: false },
        { query: "A", cached: true },
        { query: "B", cached: true },
      ],
    },
  ] as CacheTestInput[])("Retrieves data from cache (%o)", async ({ options, inputs }) => {
    vi.useFakeTimers();
    const handler = vi.fn((query) => query);
    const tool = createDummyTool(options, handler);

    for (const input of inputs) {
      if (input === null) {
        await tool.clearCache();
        continue;
      }
      if (input.sleep) {
        vi.advanceTimersByTime(input.sleep);
        continue;
      }

      await tool.run({ query: input.query }, input.options);
      if (input.cached) {
        expect(handler).not.toHaveBeenCalledOnce();
      } else {
        expect(handler).toHaveBeenCalledOnce();
      }
      handler.mockReset();
    }
  });

  describe("Tool schema", () => {
    class ComplexTool extends Tool<StringToolOutput> {
      name = "ComplexTool";
      description = "ComplexTool description";
      emitter = Emitter.root.child<any>({});

      inputSchema() {
        return z.object({ foo: z.string(), bar: z.string() });
      }

      protected async _run(arg: ToolInput<this>): Promise<StringToolOutput> {
        return new StringToolOutput(JSON.stringify(arg));
      }
    }

    it("Succeeds on valid input", async () => {
      const tool = new ComplexTool();

      const result = await tool.run({ foo: "foo", bar: "bar" });

      expect(result.getTextContent()).toMatchInlineSnapshot(`"{"foo":"foo","bar":"bar"}"`);
    });

    it("Fails on invalid input", async () => {
      const tool = new ComplexTool();

      await expect(tool.run("fail" as any)).rejects.toThrowErrorMatchingInlineSnapshot(`
        ToolInputValidationError: The received tool input does not match the expected schema.
        Input Schema: "{"type":"object","properties":{"foo":{"type":"string"},"bar":{"type":"string"}},"required":["foo","bar"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}"
        Validation Errors: [{"instancePath":"","schemaPath":"#/type","keyword":"type","params":{"type":"object"},"message":"must be object"}]
      `);
    });
  });

  describe("DynamicTool Construction", () => {
    it("Error if description is empty", async () => {
      expect(
        () =>
          new DynamicTool({
            name: "missing-desc-tool",
            description: "",
            inputSchema: z.string(),
            async handler(input) {
              return new StringToolOutput(input);
            },
          }),
      ).toThrowError(/Tool must have a description/);
    });

    it("Error if description is not defined", async () => {
      expect(
        () =>
          new DynamicTool({
            name: "missing-desc-tool",
            // @ts-expect-error intended
            description: undefined,
            inputSchema: z.string(),
            async handler(input) {
              return new StringToolOutput(input);
            },
          }),
      ).toThrowError(/Tool must have a description/);
    });

    it("Error if schema is undefined", async () => {
      expect(
        () =>
          new DynamicTool({
            name: "missing-schema-tool",
            // @ts-expect-error intended
            inputSchema: undefined,
            description: "some description",
            handler: vi.fn(),
          }),
      ).toThrowError(/Tool must have a schema/);
    });

    it("Error if name is missing", async () => {
      expect(
        () =>
          new DynamicTool({
            // @ts-expect-error intended
            name: undefined,
            inputSchema: z.string(),
            description: "some description",
            async handler(input) {
              return new StringToolOutput(input);
            },
          }),
      ).toThrowError(/Tool must have a name/);
    });

    it("Error if name does not match requirements", async () => {
      expect(
        () =>
          new DynamicTool({
            name: ";;;;;;",
            inputSchema: z.string(),
            description: "some description",
            async handler(input) {
              return new StringToolOutput(input);
            },
          }),
      ).toThrowError(/Tool name must only have -, _, letters or numbers/);
    });
  });

  describe("DynamicTool", () => {
    it("Calls provided handler", async () => {
      const handler = vi.fn((query) => query);
      const tool = new DynamicTool({
        name: "custom-tool",
        description: "Custom tool",
        inputSchema: z.string(),
        async handler(input) {
          handler(input);
          return new StringToolOutput(input);
        },
      });
      await tool.run("test");

      expect(handler).toBeCalledWith("test");
    });

    const toolDef = {
      name: "get_weather",
      description: "Determine weather in my location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state e.g. San Francisco, CA",
          },
          unit: {
            type: "string",
            enum: ["c", "f"],
          },
        },
        required: ["location"],
      },
      example: {
        location: "Prague",
        unit: "c",
      },
    };

    it("Passes handler result as the invoke fn result", async () => {
      const tool = new DynamicTool({
        name: toolDef.name,
        description: toolDef.description,
        inputSchema: toolDef.parameters,
        async handler(input) {
          const call = {
            type: "function",
            function: {
              name: this.name,
              arguments: input,
            },
          };
          return new StringToolOutput(JSON.stringify({ tool_call: call }));
        },
      });

      const result = await tool.run(toolDef.example);

      expect(result.getTextContent()).toMatchInlineSnapshot(
        `"{"tool_call":{"type":"function","function":{"name":"get_weather","arguments":{"location":"Prague","unit":"c"}}}}"`,
      );
    });

    it("Correctly transforms/parses zod schema and JsonSchema", async () => {
      const tool = new DynamicTool({
        name: "DummyTool",
        description: "DummyTool",
        inputSchema: z.object({
          a: z.string().min(1).max(100).describe("a property"),
          b: z.number().int().min(10).max(20).optional(),
          c: z.number().finite(),
          d: z.coerce.boolean().default(true),
          e: z.string().date(),
        }),
        handler: async (input) => new JSONToolOutput(input),
      });
      expect(await tool.getInputJsonSchema()).toMatchInlineSnapshot(`
        {
          "$schema": "http://json-schema.org/draft-07/schema#",
          "additionalProperties": false,
          "properties": {
            "a": {
              "description": "a property",
              "maxLength": 100,
              "minLength": 1,
              "type": "string",
            },
            "b": {
              "maximum": 20,
              "minimum": 10,
              "type": "integer",
            },
            "c": {
              "type": "number",
            },
            "d": {
              "default": true,
              "type": "boolean",
            },
            "e": {
              "format": "date",
              "type": "string",
            },
          },
          "required": [
            "a",
            "c",
            "e",
          ],
          "type": "object",
        }
      `);

      const input = {
        a: "A",
        b: 10,
        c: 15,
        d: true,
        e: "2025-01-01",
      };
      const a = await tool.run(input).then((result) => result.result);
      expect(a).toStrictEqual((await tool.inputSchema()).parse(input));
    });

    it("Emits events", async () => {
      const tool = createDummyTool({}, vi.fn().mockResolvedValue("Hey!"));
      const callbacks: any[] = [];
      tool.emitter.match("*.*", (data, event) => {
        callbacks.push({ event: event.path, data: JSON.stringify(data) });
      });
      await tool.run({ query: "Hello!" });
      expect(callbacks).toMatchInlineSnapshot(`
        [
          {
            "data": "{"input":[{"query":"Hello!"},{}]}",
            "event": "tool.dummy.run.start",
          },
          {
            "data": "{"input":{"query":"Hello!"},"options":{}}",
            "event": "tool.dummy.start",
          },
          {
            "data": "{"output":{"result":"Hey!"},"input":{"query":"Hello!"},"options":{}}",
            "event": "tool.dummy.success",
          },
          {
            "data": "null",
            "event": "tool.dummy.finish",
          },
          {
            "data": "{"result":"Hey!"}",
            "event": "tool.dummy.run.success",
          },
          {
            "data": "{"input":[{"query":"Hello!"},{}],"output":{"result":"Hey!"}}",
            "event": "tool.dummy.run.finish",
          },
        ]
      `);
    });

    it("Serializes", async () => {
      const tool = new DynamicTool({
        name: toolDef.name,
        description: toolDef.description,
        inputSchema: toolDef.parameters,
        async handler(input) {
          const call = {
            type: "function",
            function: {
              name: this.name,
              arguments: input,
            },
          };
          return new StringToolOutput(JSON.stringify({ tool_call: call }));
        },
      });

      const serialized = await tool.serialize();
      const deserialized = await DynamicTool.fromSerialized(serialized, {
        allowFunctionDeserialization: true,
      });
      verifyDeserialization(tool, deserialized);
    });
  });

  describe("Recoverable Tool Timeout", () => {
    beforeEach(() => {
      vi.useRealTimers();
    });

    it("times out and throws ToolError after deadline", async () => {
      const tool = createDummyTool({}, async () => {
        await setTimeout(200);
        return "done";
      });

      try {
        await tool.run({ query: "test" }, { timeout: 30 });
        expect.unreachable("Should have timed out");
      } catch (e) {
        expect(e).toBeInstanceOf(ToolError);
        expect((e as ToolError).message).toContain('Tool "DummyTool" timed out after 30ms');
      }
    });

    it("aborts run.signal inside _run when deadline fires", async () => {
      let seenAbortedInsideRun = false;
      class ContextSignalTool extends Tool<StringToolOutput> {
        name = "ContextSignalTool";
        description = "Testing run.signal abort";
        emitter = Emitter.root.child<any>({});

        inputSchema() {
          return z.object({ query: z.string() });
        }

        protected async _run(
          _input: ToolInput<this>,
          _options: Partial<BaseToolRunOptions>,
          run: any,
        ): Promise<StringToolOutput> {
          run.signal.addEventListener("abort", () => {
            seenAbortedInsideRun = run.signal.aborted;
          });
          await setTimeout(200);
          return new StringToolOutput("done");
        }
      }

      const tool = new ContextSignalTool({});
      try {
        await tool.run({ query: "test" }, { timeout: 30 });
        expect.unreachable("Should have timed out");
      } catch (e) {
        expect(e).toBeInstanceOf(ToolError);
        expect((e as ToolError).message).toContain('Tool "ContextSignalTool" timed out after 30ms');
      }
      expect(seenAbortedInsideRun).toBe(true);
    });

    it("caller signal abort propagates original error rather than timeout ToolError", async () => {
      const controller = new AbortController();
      const customAbortError = new Error("Caller aborted the run");

      const tool = createDummyTool({}, async () => {
        await setTimeout(200);
        return "done";
      });

      void setTimeout(20).then(() => controller.abort(customAbortError));

      try {
        await tool.run({ query: "test" }, { timeout: 500, signal: controller.signal });
        expect.unreachable("Should have aborted");
      } catch (e) {
        expect(e).toBeInstanceOf(ToolError);
        expect((e as ToolError).errors[0]).toBe(customAbortError);
        expect((e as ToolError).message).not.toContain("timed out after");
      }
    });

    it("each retry attempt gets a fresh timeout deadline", async () => {
      let attempts = 0;
      const tool = createDummyTool(
        {
          retryOptions: {
            maxRetries: 2,
            factor: 0,
          },
        },
        async () => {
          attempts++;
          if (attempts < 3) {
            await setTimeout(100);
            return "failed-too-slow";
          }
          await setTimeout(10);
          return "success-on-attempt-3";
        },
      );

      const result = await tool.run({ query: "test" }, { timeout: 40 });
      expect(result.getTextContent()).toBe("success-on-attempt-3");
      expect(attempts).toBe(3);
    });

    it("cache hit bypasses timeout and timeout is excluded from cache key", async () => {
      const handler = vi.fn().mockImplementation(async () => {
        await setTimeout(20);
        return "cached-value";
      });

      const tool = createDummyTool(
        {
          cache: new SlidingCache({ size: 10 }),
        },
        handler,
      );

      // First run populates the cache with timeout: 100ms
      const first = await tool.run({ query: "cache-query" }, { timeout: 100 });
      expect(first.getTextContent()).toBe("cached-value");
      expect(handler).toHaveBeenCalledTimes(1);

      // Second run with a super-tight timeout (1ms) should hit cache immediately without timing out
      const second = await tool.run({ query: "cache-query" }, { timeout: 1 });
      expect(second.getTextContent()).toBe("cached-value");
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it("internal tool timeout error propagates unchanged without being mislabeled", async () => {
      class InternalTimeoutError extends Error {
        name = "TimeoutError";
      }

      const tool = createDummyTool({}, async () => {
        await setTimeout(10);
        throw new InternalTimeoutError("Database connection timed out in 10ms");
      });

      try {
        await tool.run({ query: "test" }, { timeout: 1000 });
        expect.unreachable("Should have thrown internal error");
      } catch (e) {
        expect(e).toBeInstanceOf(ToolError);
        expect((e as ToolError).message).not.toContain("timed out after 1000ms");
        expect((e as ToolError).errors[0]).toBeInstanceOf(InternalTimeoutError);
        expect((e as ToolError).errors[0].message).toBe("Database connection timed out in 10ms");
      }
    });

    it.each([
      ["zero", 0],
      ["negative", -10],
      ["NaN", NaN],
      ["Infinity", Infinity],
      ["-Infinity", -Infinity],
      ["above max 32-bit signed int", 2_147_483_648],
      ["string", "invalid" as any],
    ])("rejects invalid timeout value: %s (%s)", async (_, timeout) => {
      const tool = createDummyTool({}, async () => "done");

      try {
        await tool.run({ query: "test" }, { timeout });
        expect.unreachable("Should have failed validation");
      } catch (e) {
        expect(e).toBeInstanceOf(ToolError);
        expect((e as ToolError).isRetryable).toBe(false);
        expect((e as ToolError).message).toMatch(/Property "timeout" must be a positive integer/);
      }
    });
  });
});

describe("Tool.extend metadata overrides", () => {
  const source = new DynamicTool({
    name: "OriginalTool",
    description: "Original description",
    inputSchema: z.object({ query: z.string() }),
    handler: async ({ query }) => new StringToolOutput(query),
  });

  const schema = z.object({ text: z.string() });

  it.each([
    ["no overrides", undefined, "OriginalTool", "Original description"],
    ["empty overrides", {}, "OriginalTool", "Original description"],
    ["both empty", { name: "", description: "" }, "OriginalTool", "Original description"],
    ["name only", { name: "RenamedTool" }, "RenamedTool", "Original description"],
    ["description only", { description: "New description" }, "OriginalTool", "New description"],
    [
      "both independent",
      { name: "RenamedTool", description: "New description" },
      "RenamedTool",
      "New description",
    ],
    [
      "empty description",
      { name: "RenamedTool", description: "" },
      "RenamedTool",
      "Original description",
    ],
    ["empty name", { name: "", description: "New description" }, "OriginalTool", "New description"],
  ] as const)(
    "keeps name and description independent: %s",
    async (_, overrides, name, description) => {
      const extended = source.extend(schema, ({ text }) => ({ query: text }), overrides);

      expect(extended.name).toBe(name);
      expect(extended.description).toBe(description);
      expect(extended.inputSchema()).toBe(schema);
      expect((await extended.run({ text: "local fixture" })).getTextContent()).toBe(
        "local fixture",
      );
      expect(source.name).toBe("OriginalTool");
      expect(source.description).toBe("Original description");
    },
  );
});
