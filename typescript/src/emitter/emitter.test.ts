/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { describe, expect } from "vitest";
import { EventMeta } from "@/emitter/types.js";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

describe("Emitter", () => {
  it.each([
    { blocking: false, blockingFails: false },
    { blocking: true, blockingFails: false },
    { blocking: true, blockingFails: true },
  ])("handles listener rejections while dispatching (%j)", async ({ blocking, blockingFails }) => {
    const { stdout } = await promisify(execFile)(
      process.execPath,
      [
        "--import",
        "tsx",
        "--input-type=module",
        "--eval",
        `
          import { Emitter } from ${JSON.stringify(new URL("./emitter.ts", import.meta.url).href)};
          const unhandled = [];
          process.on("unhandledRejection", (error) => unhandled.push(error.message));
          process.on("rejectionHandled", () => {});
          const emitter = new Emitter();
          emitter.on("update", async () => { throw new Error("observer failed"); }, { priority: 1 });
          emitter.on("update", async () => {
            await new Promise((resolve) => setImmediate(resolve));
            if (${blockingFails}) throw new Error("blocking listener failed");
          }, { isBlocking: ${blocking} });
          const caught = await emitter.emit("update", null).catch((error) => error.message);
          await new Promise((resolve) => setImmediate(resolve));
          emitter.destroy();
          console.log(JSON.stringify({ caught, unhandled }));
        `,
      ],
      { cwd: fileURLToPath(new URL("../..", import.meta.url)) },
    );

    expect(JSON.parse(stdout)).toEqual({
      caught: blockingFails ? "blocking listener failed" : "observer failed",
      unhandled: [],
    });
  });

  it.each([undefined, false, true])(
    "Filters descendant runs in pipe unless requested (matchNested: %s)",
    async (matchNested) => {
      const source = new Emitter({ trace: { id: "group", runId: "parent" } });
      const child = source.child({ trace: { id: "group", runId: "child", parentRunId: "parent" } });
      const target = new Emitter();
      const received: unknown[] = [];
      target.match("*.*", (data) => {
        received.push(data);
      });
      source.pipe(target, matchNested === undefined ? undefined : { matchNested });

      await source.emit("update", "parent");
      await child.emit("update", "child");

      expect(received).toEqual(matchNested ? ["parent", "child"] : ["parent"]);
    },
  );

  it("Releases a destroyed child's cleanup from its parent", () => {
    const parent = new Emitter();
    const child = parent.child();
    expect(parent.createSnapshot().cleanups).toHaveLength(1);
    child.destroy();
    child.destroy();
    expect(parent.createSnapshot().cleanups).toHaveLength(0);
  });

  it("Keeps active siblings piped when another child is destroyed", async () => {
    const parent = new Emitter();
    const first = parent.child();
    const second = parent.child();
    const listener = vitest.fn();
    parent.match("*.*", listener);
    first.destroy();
    await first.emit("event", null);
    await second.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    expect(parent.createSnapshot().cleanups).toHaveLength(1);
    second.destroy();
    expect(parent.createSnapshot().cleanups).toHaveLength(0);
  });

  it("Allows children to be destroyed after their parent", () => {
    const parent = new Emitter();
    const children = [parent.child(), parent.child()];
    parent.destroy();
    for (const child of children) {
      child.destroy();
    }
    expect(parent.createSnapshot().cleanups).toHaveLength(0);
  });

  it("Can destroy a child after a trusted serialization round trip", async () => {
    const parent = new Emitter();
    const child = parent.child();
    const restored = await Emitter.fromSerialized(await child.serialize(), {
      allowFunctionDeserialization: true,
    });
    expect(() => restored.destroy()).not.toThrow();
    child.destroy();
  });

  it.each(["g", "y", "gy"])("reuses %s regex matchers without dropping events", async (flags) => {
    const emitter = new Emitter({ namespace: ["agent"] });
    const matcher = new RegExp("^agent\\.update$", flags);
    const first = vitest.fn();
    const second = vitest.fn();
    emitter.match(matcher, first);
    emitter.match(matcher, second);

    await emitter.emit("update", 1);
    await emitter.emit("other", 0);
    await emitter.emit("update", 2);
    await emitter.emit("update", 3);

    expect(first.mock.calls.map(([data]) => data)).toEqual([1, 2, 3]);
    expect(second.mock.calls.map(([data]) => data)).toEqual([1, 2, 3]);
    expect(matcher.lastIndex).toBe(0);
    emitter.destroy();
  });

  it.each([
    { isBlocking: false, once: false },
    { isBlocking: true, once: false },
    { isBlocking: false, once: true },
    { isBlocking: true, once: true },
  ])("allows a listener to remove another listener (%j)", async (options) => {
    const emitter = new Emitter();
    const remaining = vitest.fn();
    const removed = vitest.fn();
    emitter.on("update", remaining, { priority: -1 });
    const cleanup = emitter.on("update", removed);
    const first = vitest.fn(() => cleanup());
    emitter.on("update", first, { ...options, priority: 1 });

    await emitter.emit("update", 1);
    expect(first).toHaveBeenCalledOnce();
    expect(removed).not.toHaveBeenCalled();
    expect(remaining).toHaveBeenCalledOnce();

    await emitter.emit("update", 2);
    expect(first).toHaveBeenCalledTimes(options.once ? 1 : 2);
    expect(removed).not.toHaveBeenCalled();
    expect(remaining).toHaveBeenCalledTimes(2);
    emitter.destroy();
  });

  it("Emits", async () => {
    const emitter = new Emitter({
      namespace: ["app"],
      creator: this,
    });
    const fn = vitest.fn();
    emitter.on("onStart", fn);
    await emitter.emit("onStart", "1");
    expect(fn).toBeCalledWith(
      "1",
      expect.objectContaining({
        name: "onStart",
        path: "app.onStart",
        creator: this,
        source: emitter,
        createdAt: expect.any(Date),
        context: expect.any(Object),
      }),
    );
  });

  describe("Complex workflow", () => {
    const recorder: [keyof MyEvents, unknown][] = [];

    interface MyEvents {
      start: (data: { a: number }) => void;
      success: (data: string) => void;
    }

    const emitter = new Emitter<MyEvents>({
      namespace: ["agent"],
    });

    it("Emits events in-order", async () => {
      emitter.on("start", (data) => recorder.push(["start", data]));
      emitter.on("success", (data) => recorder.push(["success", data]));
      await emitter.emit("start", { a: 1 });
      await emitter.emit("success", "a");
      await emitter.emit("success", "b");
      await emitter.emit("start", { a: 2 });
      await emitter.emit("start", { a: 3 });
      await emitter.emit("success", "c");

      expect(recorder).toMatchSnapshot();
    });

    it("Resets", async () => {
      recorder.length = 0;
      emitter.reset();
      await emitter.emit("success", "c");
      expect(recorder).toHaveLength(0);
    });
  });

  it("Accepts callbacks", async () => {
    const emitter = new Emitter<{
      start: (value: number) => void;
    }>();
    const recorder: string[] = [];
    emitter.registerCallbacks({
      start: (_, event) => recorder.push(event.path),
    });
    await emitter.emit("start", 10);
    expect(recorder).toHaveLength(1);
  });

  it("Handles nesting", async () => {
    const recorder: { source: string; meta: EventMeta }[] = [];

    const root = Emitter.root;
    root.match("*", (data, meta) => {
      recorder.push({ source: "root.*", meta });
    });
    //root.on("agent.onStart", (_, meta) =>
    //  recorder.push({ name: meta.name, path: meta.path, source: "root" }),
    //);
    //root.on("agent.runner.onStart", (_, meta) => recorder.push({ source: "root.runner", meta }));

    const agent = root.child({
      namespace: ["agent"],
    });
    agent.on("onStart", (_, meta) => recorder.push({ source: "agent", meta }));
    agent.on("*", (_, meta) => recorder.push({ source: "agent.*", meta }));
    await agent.emit("onStart", null);

    // agent.runner.onStart
    const runner = agent.child({
      namespace: ["runner"],
    });
    // agent.runner, agent.runner.onStart
    await runner.emit("onStart", null);
  });

  it("Handles priority ordering", async () => {
    const emitter = new Emitter();
    const arr: number[] = [];

    emitter.match("*.*", () => arr.push(5), { priority: 4 });
    emitter.match("*.*", () => arr.push(1), { priority: 1 });
    emitter.match("*.*", () => arr.push(2), { priority: 2 });
    emitter.match("*.*", () => arr.push(4), { priority: 3 });
    emitter.match("*.*", () => arr.push(3), { priority: 3 });
    emitter.match("*.*", () => arr.push(-1), { priority: -1 });
    emitter.match("*.*", () => arr.push(0));

    await emitter.emit("event", null);
    expect(arr).toEqual([5, 4, 3, 2, 1, 0, -1]);
  });
});
