import { Emitter, EventMeta } from "beeai-framework/emitter/emitter";

// Get the root emitter or create your own
const root = Emitter.root;

root.match("*.*", async (data: unknown, event: EventMeta) => {
  console.log(`Received event '${event.path}' with data ${JSON.stringify(data)}`);
});

await root.emit("start", { id: 123 });
await root.emit("end", { id: 123 });
