import { Emitter, EventMeta } from "beeai-framework/emitter/emitter";

const first = new Emitter({
  namespace: ["app"],
});

first.match("*.*", (data: unknown, event: EventMeta) => {
  console.log(
    `'first' has retrieved the following event ${event.path}, isDirect: ${event.source === first}`,
  );
});

const second = new Emitter({
  namespace: ["app", "llm"],
});
second.match("*.*", (data: unknown, event: EventMeta) => {
  console.log(
    `'second' has retrieved the following event '${event.path}', isDirect: ${event.source === second}`,
  );
});

// Propagate all events from the 'second' emitter to the 'first' emitter
const unpipe = second.pipe(first);

await first.emit("a", {});
await second.emit("b", {});

console.log("Unpipe");
unpipe();

await first.emit("c", {});
await second.emit("d", {});
