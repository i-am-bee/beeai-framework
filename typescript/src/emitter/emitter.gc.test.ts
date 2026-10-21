/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter } from "@/emitter/emitter.js";
import { EventMeta } from "@/emitter/types.js";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

setFlagsFromString("--expose-gc");
const gc = runInNewContext("gc") as () => void;

async function collectGarbage(until?: () => boolean) {
  for (let attempt = 0; attempt < (until ? 40 : 4); attempt++) {
    // WeakRef targets stay alive until the end of the job that dereferenced them.
    await new Promise((resolve) => setTimeout(resolve, 0));
    gc();
    // Give finalization callbacks a chance to remove the parent's bookkeeping.
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (until?.()) {
      return;
    }
  }
}

function createUnownedChild(parent: Emitter) {
  const creator: { emitter?: Emitter } = {};
  const child = parent.child({ creator });
  creator.emitter = child;
  return { child: new WeakRef(child), creator: new WeakRef(creator) };
}

function createLiveChild() {
  const parent = new Emitter({
    namespace: ["parent"],
    groupId: "group",
    context: { inherited: true },
    trace: { id: "trace", runId: "run" },
  });
  const creator = {};
  const child = parent.child({
    namespace: ["child"],
    creator,
    context: { own: true },
  });
  const events: EventMeta[] = [];
  parent.match("*.*", (_, event) => events.push(event));
  return { child, parent: new WeakRef(parent), creator, events };
}

function createLiveGrandchild(root: Emitter) {
  const middle = root.child({ namespace: ["middle"] });
  const leaf = middle.child({ namespace: ["leaf"] });
  return { leaf, middle: new WeakRef(middle) };
}

describe("Emitter garbage collection", () => {
  it("Collects unowned children and their creators while Emitter.root remains alive", async () => {
    const root = Emitter.root;
    await collectGarbage();
    const baseline = root.createSnapshot().cleanups.length;
    const refs = Array.from({ length: 10 }, () => createUnownedChild(root));

    expect(root.createSnapshot().cleanups).toHaveLength(baseline + refs.length);
    await collectGarbage(
      () =>
        refs.every((ref) => ref.child.deref() === undefined && ref.creator.deref() === undefined) &&
        root.createSnapshot().cleanups.length === baseline,
    );

    expect({
      liveChildren: refs.filter((ref) => ref.child.deref() !== undefined).length,
      liveCreators: refs.filter((ref) => ref.creator.deref() !== undefined).length,
      cleanups: root.createSnapshot().cleanups.length,
    }).toEqual({ liveChildren: 0, liveCreators: 0, cleanups: baseline });
  });

  it("Keeps a live child's parent alive and forwards the original event metadata", async () => {
    const { child, parent, creator, events } = createLiveChild();
    const localEvents: EventMeta[] = [];
    child.match("*.*", (_, event) => localEvents.push(event));

    await collectGarbage();
    expect(parent.deref()).toBeDefined();
    await child.emit("update", { value: 1 });

    expect(events).toHaveLength(1);
    expect(events[0]).toBe(localEvents[0]);
    expect(events[0]).toMatchObject({
      name: "update",
      path: "parent.child.update",
      groupId: "group",
      source: child,
      creator,
      context: { inherited: true, own: true },
      trace: { id: "trace", runId: "run" },
    });
    expect(events[0].creator).toBe(creator);
    child.destroy();
  });

  it("Preserves a live nested chain and sibling when an unowned sibling is collected", async () => {
    const root = new Emitter();
    const { leaf, middle } = createLiveGrandchild(root);
    const sibling = root.child({ namespace: ["sibling"] });
    const unowned = createUnownedChild(root);
    const paths: string[] = [];
    root.match("*.*", (_, event) => paths.push(event.path));

    await collectGarbage(
      () =>
        unowned.child.deref() === undefined &&
        unowned.creator.deref() === undefined &&
        root.createSnapshot().cleanups.length === 2,
    );
    expect(unowned.child.deref()).toBeUndefined();
    expect(unowned.creator.deref()).toBeUndefined();
    expect(root.createSnapshot().cleanups).toHaveLength(2);
    expect(middle.deref()).toBeDefined();

    await leaf.emit("update", null);
    await sibling.emit("update", null);
    expect(paths).toEqual(["middle.leaf.update", "sibling.update"]);

    leaf.destroy();
    sibling.destroy();
    root.destroy();
  });
});
