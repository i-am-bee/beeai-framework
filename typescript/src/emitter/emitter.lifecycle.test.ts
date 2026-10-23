/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { Serializer } from "@/serializer/serializer.js";
import { toBoundedFunction } from "@/serializer/utils.js";
import { describe, expect, it, vi } from "vitest";

describe("Emitter child lifecycle", () => {
  it("Detaches every child without destroying their local listeners or descendants", async () => {
    const parent = new Emitter();
    const children = [parent.child(), parent.child(), parent.child()];
    const grandchild = children[0].child();
    const local = vi.fn();
    children[0].match("*.*", local);
    parent.destroy();
    const upstream = vi.fn();
    parent.match("*.*", upstream);

    for (const child of children) {
      await child.emit("event", null);
    }
    await grandchild.emit("event", null);
    expect(upstream).not.toHaveBeenCalled();
    expect(local).toHaveBeenCalledTimes(2);
    expect(children[0].createSnapshot().cleanups).toHaveLength(1);
    children.forEach((child) => child.destroy());
    grandchild.destroy();
  });

  it("Keeps reset pipes persistent and detaches only the automatic pipe", async () => {
    const parent = new Emitter();
    const child = parent.child();
    const stopManualPipe = child.pipe(parent);
    child.reset();
    parent.destroy();
    const listener = vi.fn();
    parent.match("*.*", listener);
    await child.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    stopManualPipe();
    await child.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    child.destroy();
    parent.destroy();
  });

  it("Clones a child with an independent parent link", async () => {
    const parent = new Emitter();
    const child = parent.child();
    const cloned = await child.clone();
    const listener = vi.fn();
    parent.match("*.*", listener);
    expect(parent.createSnapshot().cleanups).toHaveLength(2);
    cloned.destroy();
    await child.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    expect(parent.createSnapshot().cleanups).toHaveLength(1);
    child.destroy();
    parent.destroy();
  });

  it("Clones a parent without taking ownership of the original children", async () => {
    const parent = new Emitter();
    const child = parent.child();
    const cloned = await parent.clone();
    const listener = vi.fn();
    parent.match("*.*", listener);
    cloned.destroy();
    await child.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    parent.destroy();
    child.destroy();
  });

  it("Detaches old automatic relationships when loading into an existing emitter", async () => {
    const parent = new Emitter();
    const child = parent.child();
    parent.loadSnapshot(new Emitter().createSnapshot());
    const listener = vi.fn();
    parent.match("*.*", listener);
    await child.emit("event", null);
    expect(listener).not.toHaveBeenCalled();
    child.destroy();
    parent.destroy();
  });

  it("Loads snapshots created before parent relationship metadata was added", async () => {
    const original = new Emitter({ namespace: ["legacy"] });
    const listener = vi.fn();
    original.match("*.*", listener);
    const {
      parent: _parent,
      childCleanupIndices: _childCleanupIndices,
      ...legacy
    } = original.createSnapshot();
    const restored = new Emitter();
    restored.loadSnapshot(legacy);
    await restored.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    restored.destroy();
    original.destroy();
  });

  it("Restores a serialized child with a working automatic parent cleanup", async () => {
    const parent = new Emitter();
    const child = parent.child();
    const restored = await Emitter.fromSerialized<Emitter>(await child.serialize(), {
      allowFunctionDeserialization: true,
    });
    const restoredParent = restored.createSnapshot().parent!;
    const listener = vi.fn();
    restoredParent.match("*.*", listener);
    await restored.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    restoredParent.destroy();
    restoredParent.match("*.*", listener);
    await restored.emit("event", null);
    expect(listener).toHaveBeenCalledOnce();
    restored.destroy();
    parent.destroy();
    child.destroy();
  });

  it.each([false, true])(
    "Restores a live parent pipe and its cleanup (child first: %s)",
    async (childFirst) => {
      const parent = new Emitter();
      const child = parent.child();
      const seen: string[] = [];
      parent.match(
        "*.*",
        toBoundedFunction((value: string) => seen.push(value), [{ value: seen, name: "seen" }]),
      );
      const pair = childFirst ? { child, parent, seen } : { parent, child, seen };
      const restored = await Serializer.deserialize<typeof pair>(
        await Serializer.serialize(pair),
        undefined,
        false,
        {
          allowFunctionDeserialization: true,
        },
      );

      expect(restored.parent.createSnapshot().cleanups).toHaveLength(1);
      await restored.child.emit("event", "before");
      expect(restored.seen).toEqual(["before"]);
      restored.parent.destroy();
      const after = vi.fn();
      restored.parent.match("*.*", after);
      await restored.child.emit("event", "after");
      expect(after).not.toHaveBeenCalled();
      restored.child.destroy();
      parent.destroy();
      child.destroy();
    },
  );

  it("Restores parent placeholders referenced by their context", async () => {
    const context: { child?: Emitter } = {};
    const parent = new Emitter({ context });
    const child = parent.child();
    context.child = child;
    const restored = await Serializer.deserialize<{ parent: Emitter; child: Emitter }>(
      await Serializer.serialize({ parent, child }),
      undefined,
      false,
      { allowFunctionDeserialization: true },
    );
    expect((restored.parent.context as typeof context).child).toBe(restored.child);
    expect(restored.parent.createSnapshot().cleanups).toHaveLength(1);
    const upstream = vi.fn();
    restored.parent.match("*.*", upstream);
    await restored.child.emit("event", null);
    expect(upstream).toHaveBeenCalledOnce();
    restored.parent.destroy();
    restored.parent.match("*.*", upstream);
    await restored.child.emit("event", null);
    expect(upstream).toHaveBeenCalledOnce();
    restored.child.destroy();
    parent.destroy();
    child.destroy();
  });
});
