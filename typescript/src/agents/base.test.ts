/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { setImmediate } from "node:timers/promises";
import { BaseAgent, AgentError } from "@/agents/base.js";
import { Emitter } from "@/emitter/emitter.js";
import { UnconstrainedMemory } from "@/memory/unconstrainedMemory.js";
import { vi } from "vitest";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

class TestAgent extends BaseAgent<string, string> {
  memory = new UnconstrainedMemory();
  readonly emitter = Emitter.root.child({ namespace: ["agent", "test"], creator: this });

  constructor(private readonly handler: (input: string) => Promise<string>) {
    super();
  }

  protected async _run(input: string) {
    return this.handler(input);
  }
}

describe("BaseAgent", () => {
  it("rejects a competing lazy run without releasing the active run", async () => {
    const started = deferred();
    const release = deferred();
    const handler = vi.fn(async (input: string) => {
      started.resolve();
      await release.promise;
      return input;
    });
    const agent = new TestAgent(handler);
    const first = agent.run("first");
    const second = agent.run("second");
    const results = Promise.allSettled([first, second]);

    try {
      await started.promise;
      await setImmediate();
      expect(handler).toHaveBeenCalledTimes(1);
      expect(handler).toHaveBeenCalledWith("first");
      expect(() => agent.run("third")).toThrow("Agent is already running!");
    } finally {
      release.resolve();
      await results;
      agent.destroy();
    }

    const [firstResult, secondResult] = await results;
    expect(firstResult).toEqual({ status: "fulfilled", value: "first" });
    expect(secondResult.status).toBe("rejected");
    if (secondResult.status === "rejected") {
      expect(secondResult.reason).toBeInstanceOf(AgentError);
      expect(secondResult.reason.message).toBe("Agent is already running!");
    }
  });

  it("rejects a new run while execution is already active", async () => {
    const started = deferred();
    const release = deferred();
    const agent = new TestAgent(async (input) => {
      started.resolve();
      await release.promise;
      return input;
    });
    const running = Promise.resolve(agent.run("first"));
    try {
      await started.promise;
      expect(() => agent.run("second")).toThrow("Agent is already running!");
    } finally {
      release.resolve();
      await running;
      agent.destroy();
    }
  });

  it.each([false, true])("allows a later run after failure=%s", async (fail) => {
    const handler = vi.fn(async (input: string) => input);
    if (fail) {
      handler.mockRejectedValueOnce(new Error("First run failed"));
    }
    const agent = new TestAgent(handler);
    try {
      if (fail) {
        await expect(agent.run("first")).rejects.toBeInstanceOf(AgentError);
      } else {
        await expect(agent.run("first")).resolves.toBe("first");
      }
      await expect(agent.run("second")).resolves.toBe("second");
      expect(handler).toHaveBeenCalledTimes(2);
    } finally {
      agent.destroy();
    }
  });
});
