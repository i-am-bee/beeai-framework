/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { describe, expect } from "vitest";
import { EventMeta } from "@/emitter/types.js";

describe("Emitter", () => {
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
