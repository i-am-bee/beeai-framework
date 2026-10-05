/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { setTimeout as delay } from "node:timers/promises";
import { describe, expect, it, vi } from "vitest";
import { BaseAgent } from "@/agents/base.js";
import { RunContext, RunContextCallbacks } from "@/context.js";
import { Emitter } from "@/emitter/emitter.js";
import { InferCallbackValue } from "@/emitter/types.js";
import { FrameworkError } from "@/errors.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";

describe("RunContext cancellation", () => {
  it.each(["input", "before await", "middleware", "start listener"])(
    "Rejects cancellation from %s before calling the handler",
    async (mode) => {
      const controller = new AbortController();
      const reason = new Error("Run cancelled");
      const instance = { emitter: new Emitter() };
      const events: { name: string; data: unknown }[] = [];
      const handler = vi.fn(async () => "result");
      let context: RunContext<typeof instance> | undefined;
      if (mode === "input") {
        controller.abort(reason);
      }

      const run = RunContext.enter(
        instance,
        { signal: controller.signal, params: ["input"] },
        handler,
      )
        .middleware((ctx) => {
          context = ctx;
          ctx.runParams = ["middleware input"];
          if (mode === "middleware") {
            ctx.abort(reason);
          }
        })
        .observe((emitter) => {
          emitter.match("*.*", (data, event) => events.push({ name: event.name, data }));
          emitter.match<InferCallbackValue<RunContextCallbacks["start"]>>(
            (event) => event.name === "start",
            async (data) => {
              expect(data.input).toEqual(["middleware input"]);
              await Promise.resolve();
              data.input = ["modified input"];
              if (mode === "start listener") {
                controller.abort(reason);
              }
            },
          );
        });
      if (mode === "before await") {
        controller.abort(reason);
      }

      try {
        const error = await run.catch((error: FrameworkError) => error);
        expect(error).toBeInstanceOf(FrameworkError);
        expect((error as FrameworkError).getCause()).toBe(reason);
        expect(handler).not.toHaveBeenCalled();
        expect(events.map((event) => event.name)).toEqual(["start", "error", "finish"]);
        expect(events[1].data).toBe(error);
        expect(events[2].data).toEqual({ input: ["modified input"], output: error });
        expect(context?.runParams).toEqual(["modified input"]);
        expect(context?.signal.aborted).toBe(true);
        expect(context?.signal.reason).toBe(reason);
      } finally {
        instance.emitter.destroy();
      }
    },
  );

  it("Rejects a child run that inherits its parent's cancelled signal", async () => {
    const parent = { emitter: new Emitter() };
    const child = { emitter: new Emitter() };
    const reason = new Error("Parent run cancelled");
    const handler = vi.fn(async () => "child result");
    const run = RunContext.enter(parent, { params: [] }, async (context) => {
      context.abort(reason);
      return await RunContext.enter(child, { params: [] }, handler);
    });
    try {
      const error = await run.catch((error: FrameworkError) => error);
      expect(error).toBeInstanceOf(FrameworkError);
      expect((error as FrameworkError).getCause()).toBe(reason);
      expect(handler).not.toHaveBeenCalled();
    } finally {
      child.emitter.destroy();
      parent.emitter.destroy();
    }
  });

  it("Preserves input changes and success events for a normal run", async () => {
    const instance = { emitter: new Emitter() };
    const events: string[] = [];
    let context: RunContext<typeof instance> | undefined;
    const run = RunContext.enter(instance, { params: ["input"] }, async (ctx) => {
      context = ctx;
      expect(ctx.runParams).toEqual(["modified input"]);
      return "result";
    }).observe((emitter) => {
      emitter.match("*.*", (_, event) => events.push(event.name));
      emitter.match<InferCallbackValue<RunContextCallbacks["start"]>>(
        (event) => event.name === "start",
        (data) => {
          data.input = ["modified input"];
        },
      );
    });
    try {
      await expect(run).resolves.toBe("result");
      expect(events).toEqual(["start", "success", "finish"]);
      expect(context?.signal.aborted).toBe(true);
      expect(context?.signal.reason).toBeInstanceOf(FrameworkError);
    } finally {
      instance.emitter.destroy();
    }
  });

  it("Preserves a result supplied by a start listener", async () => {
    const instance = { emitter: new Emitter() };
    const handler = vi.fn(async () => "handler result");
    const run = RunContext.enter(instance, { params: [] }, handler).observe((emitter) => {
      emitter.match<InferCallbackValue<RunContextCallbacks["start"]>>(
        (event) => event.name === "start",
        (data) => {
          data.output = "supplied result";
        },
      );
    });
    try {
      await expect(run).resolves.toBe("supplied result");
      expect(handler).not.toHaveBeenCalled();
    } finally {
      instance.emitter.destroy();
    }
  });

  it("Preserves cancellation after the handler has started", async () => {
    const controller = new AbortController();
    const reason = new Error("Active run cancelled");
    const instance = { emitter: new Emitter() };
    const events: string[] = [];
    let started = false;
    const handler = async () => {
      started = true;
      setTimeout(() => controller.abort(reason), 0);
      await delay(30);
      return "result";
    };
    const run = RunContext.enter(
      instance,
      { signal: controller.signal, params: [] },
      handler,
    ).observe((emitter) => emitter.match("*.*", (_, event) => events.push(event.name)));
    try {
      const error = await run.catch((error: FrameworkError) => error);
      expect(error).toBeInstanceOf(FrameworkError);
      expect((error as FrameworkError).getCause()).toBe(reason);
      expect(started).toBe(true);
      expect(events).toEqual(["start", "error", "finish"]);
      await delay(40);
      expect(events).toEqual(["start", "error", "finish"]);
    } finally {
      instance.emitter.destroy();
    }
  });

  it("Rejects a pre-cancelled public agent run without invoking the agent", async () => {
    class TestAgent extends BaseAgent<string, string> {
      memory = new UnconstrainedMemory();
      emitter = new Emitter();
      protected _run = vi.fn(async () => "result");
      get started() {
        return this._run.mock.calls.length;
      }
    }
    const agent = new TestAgent();
    const controller = new AbortController();
    const reason = new Error("Agent run cancelled");
    controller.abort(reason);
    try {
      const error = await agent
        .run("input", { signal: controller.signal })
        .catch((error: FrameworkError) => error);
      expect(error).toBeInstanceOf(FrameworkError);
      expect((error as FrameworkError).getCause()).toBe(reason);
      expect(agent.started).toBe(0);
      await expect(agent.run("input")).resolves.toBe("result");
      expect(agent.started).toBe(1);
    } finally {
      agent.destroy();
    }
  });
});
