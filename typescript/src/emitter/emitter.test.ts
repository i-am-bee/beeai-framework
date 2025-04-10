/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Emitter } from "@/emitter/emitter.js";
import { describe, expect } from "vitest";
import { EventMeta } from "@/emitter/types.js";

describe("Emitter", () => {
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
});
