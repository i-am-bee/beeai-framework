/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Emitter, EventMeta } from "beeai-framework/emitter/emitter";

// Get the root emitter or create your own
const root = Emitter.root;

root.match("*.*", async (data: unknown, event: EventMeta) => {
  console.log(`Received event '${event.path}' with data ${JSON.stringify(data)}`);
});

await root.emit("start", { id: 123 });
await root.emit("end", { id: 123 });
