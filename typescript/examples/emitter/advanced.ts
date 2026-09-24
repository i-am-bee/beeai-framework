import { Emitter, EventMeta, Callback } from "beeai-framework/emitter/emitter";

// Define events in advanced
interface Events {
  start: Callback<{ id: number }>;
  update: Callback<{ id: number; data: string }>;
}

// Create emitter with a type support
const emitter = Emitter.root.child<Events>({
  namespace: ["bee", "demo"],
  creator: {}, // typically a class
  context: {}, // custom data (propagates to the event's context property)
  groupId: undefined, // optional id for grouping common events (propagates to the event's groupId property)
  trace: undefined, // data related to identity what emitted what and which context (internally used by framework's components)
});

// Listen for "start" event
emitter.on("start", async (data, event: EventMeta) => {
  console.log(`Received ${event.name} event with id "${data.id}"`);
});

// Listen for "update" event
emitter.on("update", async (data, event: EventMeta) => {
  console.log(`Received ${event.name}' with id "${data.id}" and data ${data.data}`);
});

await emitter.emit("start", { id: 123 });
await emitter.emit("update", { id: 123, data: "Hello Bee!" });
