# Migration Guide

## 0.0.X -> 0.1.0 (2025-02-11)

### Summary

- `ChatLLM` class was replaced by `ChatModel` class and embedding functionality has been replaced by `EmbeddingModel` class.
- `BaseMessage` class was replaced by `Message` and its subtypes (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolMessage`).
- `TokenMemory` no longer uses `LLM` instance to infer `maxTokens`, user needs to do that manually (if needed).
- Tokenization has been removed.
- Non-Chat LLM class (`LLM`) has been removed.
- The`IBMvLLM` adapter has been removed.
- Parsers were moved from `beeai-framework/agents/parsers` to `beeai-framework/parsers`.
- Workflows are no longer experimental and were moved from `beeai-framework/experimental/workflows` to `beeai-framework/workflows`.
- LLM Drivers concept has been replaced by Structured Outputs via `model.createStructure` method.

### Models

#### ❌ Old Way

```ts
import { OllamaChatLLM } from "beeai-framework/adapters/ollama/chat";
import { BaseMessage } from "beeai-framework/llms/primitives/message";

const model = new OllamaChatLLM({
  modelId: "llama3.1",
});

const response = await model.generate(
  [
    BaseMessage.of({
      role: "user",
      text: "Hello Bee!",
    }),
  ],
  {
    stream: true,
  },
);
console.log(response.getTextContent());
```

#### ✅ New Way

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";

const model = await ChatModel.fromName("ollama:llama3.1");
const response = await model.create({
  messages: [new UserMessage("Hello Bee!")],
});
console.log(response.getTextContent());
```

or you can initiate the provider directly

```ts
import { OllamaChatModel } from "beeai-framework/adapters/ollama/chat";
import { UserMessage } from "beeai-framework/backend/core";

const model = new OllamaChatModel("llama3.1");
const response = await model.create({
  messages: [new UserMessage("Hello Bee!")],
});
console.log(response.getTextContent());
```

More examples can be found in [Backend Documentation Page](/typescript/docs/backend.md).

### Messages

The `BaseMessage` class was replaced by `Message` and its subtypes (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolMessage`).

#### ❌ Old Way

```ts
import { BaseMessage } from "beeai-framework/llms/primitives/message";
const a = BaseMessage.of({ role: "user", text: "hello", meta: { createdAt: new Date() } });
```

#### ✅ New Way

```ts
import { Message } from "beeai-framework/backend/core";

const a = Message.of({ role: "user", text: "Hello", meta: { createdAt: new Date() } });
```

The new version supports more complex content types.

```ts
import { UserMessage } from "beeai-framework/backend/core";

// using a factory function
const msg = new UserMessage({
  type: "file",
  data: await fs.promises.readFile("document.txt"),
  mimeType: "text/plain",
});
```

```ts
import { UserMessage } from "beeai-framework/backend/core";

// using a factory function
const msg = new UserMessage({
  type: "file",
  data: await fs.promises.readFile("document.txt"),
  mimeType: "text/plain",
});
```

```ts
import { UserMessage, AssistantMessage, SystemMessage } from "beeai-framework/backend/core";
const a = new UserMessage("Hello assistant!");
const b = new AssistantMessage("Hello user!");
const c = new SystemMessage("You are a helpful assistant.");
```

More examples can be found in [Backend Documentation Page](/typescript/docs/backend.md).

### Serialization

The following methods present in `Serializable` class are now asynchronous.

- `serialize`
- `deserialize`
- `createSnapshot`
- `loadSnapshot`

The same applies to the following static methods.

- `fromSerialized`
- `fromSnapshot`

#### ❌ Old Way

```ts
import { TokenMemory } from "beeai-framework/memory/tokenMemory";

const a = new TokenMemory();
const json = a.serialize();

const b = TokenMemory.fromSerialized(json);
```

#### ✅ New Way

```ts
import { TokenMemory } from "beeai-framework/memory/tokenMemory";

const a = new TokenMemory();
const json = await a.serialize();

const b = await TokenMemory.fromSerialized(json);
```

### Workflows

#### ❌ Old Way

```ts
import { Workflow } from "beeai-framework/experimental/workflows/workflow";

const schema = z.object({
  value: z.number(),
  hops: z.number().default(0),
});

const workflow = new Workflow({ schema })
  .addStep("router", async (state) => {
    if (state.hops > 20) {
      return { next: Workflow.END };
    }
    return {
      update: { hops: state.hops + 1 },
      next: Math.random() > 0.5 ? "increment" : "decrement",
    };
  })
  .addStep("decrement", async (state) => {
    return {
      update: { hops: state.hops + 1, value: state.value - 1 },
      next: "router",
    };
  })
  .addStep("increment", async (state) => {
    return {
      update: { hops: state.hops + 1, value: state.value + 1 },
      next: "router",
    };
  });
```

#### ✅ New Way

```ts
import { Workflow } from "beeai-framework/workflows/workflow";

const schema = z.object({
  value: z.number(),
  hops: z.number().default(0),
});

const workflow = new Workflow({ schema })
  .addStep("router", async (state) => {
    if (state.hops > 20) {
      return Workflow.END;
    }
    state.hops += 1;
    return Math.random() > 0.5 ? "increment" : "decrement";
  })
  .addStep("decrement", async (state) => {
    state.hops += 1;
    state.value -= 1;
    return "router";
  })
  .addStep("increment", async (state) => {
    state.hops += 1;
    state.value += 1;
    return "router";
  });
```

### LLM Drivers

#### ✅ New Way

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";

const model = await ChatModel.fromName("ollama:llama3.1");
const { object } = await model.createStructure({
  schema: { answer: z.string() },
  messages: [new UserMessage("What has keys but can’t open locks?")],
  maxRetries: 3,
});
console.log(`Answer: ${object.answer}`);
```

#### ❌ Old Way

```ts
import { OllamaChatLLM } from "beeai-framework/adapters/ollama/chat";
import { JsonDriver } from "beeai-framework/llms/drivers/json";
import { BaseMessage } from "beeai-framework/llms/primitives/message";

const llm = new OllamaChatLLM({ modelId: "llama3.1" });
const driver = new JsonDriver(llm);

const { parsed } = await driver.generate(
  z.object({ name: z.string() }),
  [BaseMessage.of({ role: "user", text: "What has keys but can’t open locks?" })],
  { maxRetries: 3 },
);
console.log(`Answer: ${parsed.answer}`);
```
