# Emitter (Observability)

> Location within the framework `beeai-framework/emitter`.

An emitter is a core functionality of the framework that allows you to see what is happening under the hood.

## Standalone usage

The following examples demonstrate how [`Emitter`](/typescript/src/emitter/emitter.ts) concept works.

### Basic Usage

<!-- embedme examples/emitter/base.ts -->

```ts
import { Emitter, EventMeta } from "beeai-framework/emitter/emitter";

// Get the root emitter or create your own
const root = Emitter.root;

root.match("*.*", async (data: unknown, event: EventMeta) => {
  console.log(`Received event '${event.path}' with data ${JSON.stringify(data)}`);
});

await root.emit("start", { id: 123 });
await root.emit("end", { id: 123 });
```

_Source: [examples/emitter/base.ts](/typescript/examples/emitter/base.ts)_

> [!NOTE]
>
> You can create your own emitter by initiating the `Emitter` class, but typically it's better to use or fork the root one (as can be seen in the following examples).

### Advanced

<!-- embedme examples/emitter/advanced.ts -->

```ts
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
```

_Source: [examples/emitter/advanced.ts](/typescript/examples/emitter/advanced.ts)_

> [!NOTE]
>
> Because we created the `Emitter` instance directly emitted events will not be propagated to the `root` which may or may not be desired.
> The `piping` concept is explained later on.

### Event Matching

<!-- embedme examples/emitter/matchers.ts -->

```ts
import { Callback, Emitter } from "beeai-framework/emitter/emitter";
import { ChatModel } from "beeai-framework/backend/chat";

interface Events {
  update: Callback<{ data: string }>;
}

const emitter = new Emitter<Events>({
  namespace: ["app"],
});

// Match events by a concrete name (strictly typed)
emitter.on("update", async (data, event) => {});

// Match all events emitted directly on the instance (not nested)
emitter.match("*", async (data, event) => {});

// Match all events (included nested)
emitter.match("*.*", async (data, event) => {});

// Match events by providing a filter function
emitter.match(
  (event) => event.creator instanceof ChatModel,
  async (data, event) => {},
);

// Match events by regex
emitter.match(/watsonx/, async (data, event) => {});
```

_Source: [examples/emitter/matchers.ts](/typescript/examples/emitter/matchers.ts)_

### Event Piping

<!-- embedme examples/emitter/piping.ts -->

```ts
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
```

_Source: [examples/emitter/piping.ts](/typescript/examples/emitter/piping.ts)_

## Framework Usage

Typically, you consume out-of-the-box modules that use the `Emitter` concept on your behalf.

## Agent usage

<!-- embedme examples/emitter/agentMatchers.ts -->

```ts
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  memory: new UnconstrainedMemory(),
  tools: [],
});

// Matching events on the instance level
agent.emitter.match("*.*", (data, event) => {});

await agent
  .run({
    prompt: "Hello agent!",
  })
  .observe((emitter) => {
    // Matching events on the execution (run) level
    emitter.match("*.*", (data, event) => {
      console.info(`RUN LOG: received event '${event.path}'`);
    });
  });
```

_Source: [examples/emitter/agentMatchers.ts](/typescript/examples/emitter/agentMatchers.ts)_

> [!IMPORTANT]
>
> The `observe` method is also supported on [`Tools`](./tools.md) and [`Backend`](./backend.md).

> [!TIP]
>
> The more complex agentic example can be found [here](/typescript/examples/agents/react.ts).

> [!TIP]
>
> To verify if a given class instance has one, check for the presence of the `emitter` property.
