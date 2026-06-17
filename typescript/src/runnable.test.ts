/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { Emitter } from "@/emitter/emitter.js";
import { AssistantMessage, UserMessage } from "@/backend/message.js";
import { GetRunContext, MiddlewareType } from "@/context.js";
import { Runnable, RunnableOptions, RunnableOutput, runnableEntry } from "@/runnable.js";

/**
 * A minimal concrete Runnable used for testing: echoes the input messages
 * back as output, wrapped in an AssistantMessage summary.
 */
class EchoRunnable extends Runnable<RunnableOutput> {
  protected readonly _emitter = Emitter.root.child<any>({ namespace: ["test", "echo"] });

  get emitter() {
    return this._emitter;
  }

  run(input: UserMessage[], options?: RunnableOptions) {
    return runnableEntry(this, input, options, async (_context, runInput) => {
      return new RunnableOutput({
        output: [
          ...runInput,
          new AssistantMessage(`echoed ${runInput.length} message(s)`),
        ],
      });
    });
  }
}

describe("Runnable", () => {
  it("runs and returns a RunnableOutput", async () => {
    const runnable = new EchoRunnable();
    const input = [new UserMessage("hello")];

    const result = await runnable.run(input);

    expect(result).toBeInstanceOf(RunnableOutput);
    expect(result.output).toHaveLength(2);
    expect(result.lastMessage.text).toBe("echoed 1 message(s)");
  });

  it("falls back to an empty AssistantMessage when output is empty", () => {
    const output = new RunnableOutput({ output: [] });

    expect(output.lastMessage).toBeInstanceOf(AssistantMessage);
    expect(output.lastMessage.text).toBe("");
  });

  it("lets middleware mutate the input before the handler runs", async () => {
    // Regression test: the handler must read the (possibly mutated)
    // input from `context.runParams`, not from the original closured
    // argument, otherwise middleware rewrites are silently dropped.
    const rewrittenInput = [new UserMessage("rewritten by middleware")];

    const mutateInputMiddleware: MiddlewareType<EchoRunnable> = (
      context: GetRunContext<EchoRunnable>,
    ) => {
      context.runParams = rewrittenInput;
    };

    const runnable = new EchoRunnable([mutateInputMiddleware]);
    const originalInput = [new UserMessage("original")];

    const result = await runnable.run(originalInput);

    // The handler should have seen the middleware's rewritten input,
    // not the original argument passed into `run()`.
    expect(result.output[0].text).toBe("rewritten by middleware");
    expect(result.lastMessage.text).toBe("echoed 1 message(s)");
  });
});