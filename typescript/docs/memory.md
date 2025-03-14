# Memory

> [!TIP]
>
> Location within the framework `beeai-framework/memory`.

Memory in the context of an agent refers to the system's capability to store, recall, and utilize information from past interactions. This enables the agent to maintain context over time, improve its responses based on previous exchanges, and provide a more personalized experience.

## Usage

### Capabilities showcase

<!-- embedme examples/memory/base.ts -->

```ts
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { AssistantMessage, SystemMessage, UserMessage } from "beeai-framework/backend/message";

const memory = new UnconstrainedMemory();

// Single message
await memory.add(new SystemMessage(`You are a helpful assistant.`));

// Multiple messages
await memory.addMany([new UserMessage(`What can you do?`), new AssistantMessage(`Everything!`)]);

console.info(memory.isEmpty()); // false
console.info(memory.messages); // prints all saved messages
console.info(memory.asReadOnly()); // returns a NEW read only instance
memory.reset(); // removes all messages
```

_Source: [examples/memory/base.ts](/typescript/examples/memory/base.ts)_

### Usage with LLMs

<!-- embedme examples/memory/llmMemory.ts -->

```ts
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Message } from "beeai-framework/backend/message";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const memory = new UnconstrainedMemory();
await memory.addMany([
  Message.of({
    role: "system",
    text: `Always respond very concisely.`,
  }),
  Message.of({ role: "user", text: `Give me first 5 prime numbers.` }),
]);

// Generate response
const llm = new OllamaChatModel("llama3.1");
const response = await llm.create({ messages: memory.messages });
await memory.add(Message.of({ role: "assistant", text: response.getTextContent() }));

console.log(`Conversation history`);
for (const message of memory) {
  console.log(`${message.role}: ${message.text}`);
}
```

_Source: [examples/memory/llmMemory.ts](/typescript/examples/memory/llmMemory.ts)_

> [!TIP]
>
> Memory for non-chat LLMs works exactly the same way.

### Usage with agents

<!-- embedme examples/memory/agentMemory.ts -->

```ts
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const agent = new ReActAgent({
  memory: new UnconstrainedMemory(),
  llm: new OllamaChatModel("llama3.1"),
  tools: [],
});
await agent.run({ prompt: "Hello world!" });

console.info(agent.memory.messages.length); // 2

const userMessage = agent.memory.messages[0];
console.info(`User: ${userMessage.text}`); // User: Hello world!

const agentMessage = agent.memory.messages[1];
console.info(`Agent: ${agentMessage.text}`); // Agent: Hello! It's nice to chat with you.
```

_Source: [examples/memory/agentMemory.ts](/typescript/examples/memory/agentMemory.ts)_

> [!TIP]
>
> If your memory already contains the user message, run the agent with `prompt: null`.

> [!NOTE]
>
> ReAct Agent internally uses `TokenMemory` to store intermediate steps for a given run.

> [!NOTE]
>
> Agent typically works with a memory similar to what was just shown.

## Memory types

The framework provides multiple out-of-the-box memory implementations.

### UnconstrainedMemory

Unlimited in size.

<!-- embedme examples/memory/unconstrainedMemory.ts -->

```ts
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Message } from "beeai-framework/backend/message";

const memory = new UnconstrainedMemory();
await memory.add(
  Message.of({
    role: "user",
    text: `Hello world!`,
  }),
);

console.info(memory.isEmpty()); // false
console.log(memory.messages.length); // 1
console.log(memory.messages);
```

_Source: [examples/memory/unconstrainedMemory.ts](/typescript/examples/memory/unconstrainedMemory.ts)_

### SlidingMemory

Keeps last `k` entries in the memory. The oldest ones are deleted (unless specified otherwise).

<!-- embedme examples/memory/slidingMemory.ts -->

```ts
import { SlidingMemory } from "beeai-framework/memory/slidingMemory";
import { Message } from "beeai-framework/backend/message";

const memory = new SlidingMemory({
  size: 3, // (required) number of messages that can be in the memory at a single moment
  handlers: {
    // optional
    // we select a first non-system message (default behaviour is to select the oldest one)
    removalSelector: (messages) => messages.find((msg) => msg.role !== "system")!,
  },
});

await memory.add(Message.of({ role: "system", text: "You are a guide through France." }));
await memory.add(Message.of({ role: "user", text: "What is the capital?" }));
await memory.add(Message.of({ role: "assistant", text: "Paris" }));
await memory.add(Message.of({ role: "user", text: "What language is spoken there?" })); // removes the first user's message
await memory.add(Message.of({ role: "assistant", text: "French" })); // removes the first assistant's message

console.info(memory.isEmpty()); // false
console.log(memory.messages.length); // 3
console.log(memory.messages);
```

_Source: [examples/memory/slidingMemory.ts](/typescript/examples/memory/slidingMemory.ts)_

### TokenMemory

Ensures that the token sum of all messages is below the given threshold.
If overflow occurs, the oldest message will be removed.

<!-- embedme examples/memory/tokenMemory.ts -->

```ts
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { Message } from "beeai-framework/backend/message";

const memory = new TokenMemory({
  maxTokens: undefined, // optional (default is 128k),
  capacityThreshold: 0.75, // maxTokens*capacityThreshold = threshold where we start removing old messages
  syncThreshold: 0.25, // maxTokens*syncThreshold = threshold where we start to use a real tokenization endpoint instead of guessing the number of tokens
  handlers: {
    // optional way to define which message should be deleted (default is the oldest one)
    removalSelector: (messages) => messages.find((msg) => msg.role !== "system")!,

    // optional way to estimate the number of tokens in a message before we use the actual tokenize endpoint (number of tokens < maxTokens*syncThreshold)
    estimate: (msg) => Math.ceil((msg.role.length + msg.text.length) / 4),
  },
});

await memory.add(Message.of({ role: "system", text: "You are a helpful assistant." }));
await memory.add(Message.of({ role: "user", text: "Hello world!" }));

console.info(memory.isDirty); // is the consumed token count estimated or retrieved via the tokenize endpoint?
console.log(memory.tokensUsed); // number of used tokens
console.log(memory.stats()); // prints statistics
await memory.sync(); // calculates real token usage for all messages marked as "dirty"
```

_Source: [examples/memory/tokenMemory.ts](/typescript/examples/memory/tokenMemory.ts)_

### SummarizeMemory

Only a single summarization of the conversation is preserved. Summarization is updated with every new message.

<!-- embedme examples/memory/summarizeMemory.ts -->

```ts
import { Message } from "beeai-framework/backend/message";
import { SummarizeMemory } from "beeai-framework/memory/summarizeMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const memory = new SummarizeMemory({
  llm: new OllamaChatModel("llama3.1"),
});

await memory.addMany([
  Message.of({ role: "system", text: "You are a guide through France." }),
  Message.of({ role: "user", text: "What is the capital?" }),
  Message.of({ role: "assistant", text: "Paris" }),
  Message.of({ role: "user", text: "What language is spoken there?" }),
]);

console.info(memory.isEmpty()); // false
console.log(memory.messages.length); // 1
console.log(memory.messages[0].text); // The capital city of France is Paris, ...
```

_Source: [examples/memory/summarizeMemory.ts](/typescript/examples/memory/summarizeMemory.ts)_

## Creating a custom memory provider

To create your memory implementation, you must implement the `BaseMemory` class.

<!-- embedme examples/memory/custom.ts -->

```ts
import { BaseMemory } from "beeai-framework/memory/base";
import { Message } from "beeai-framework/backend/message";
import { NotImplementedError } from "beeai-framework/errors";

export class MyMemory extends BaseMemory {
  get messages(): readonly Message[] {
    throw new NotImplementedError("Method not implemented.");
  }

  add(message: Message, index?: number): Promise<void> {
    throw new NotImplementedError("Method not implemented.");
  }

  delete(message: Message): Promise<boolean> {
    throw new NotImplementedError("Method not implemented.");
  }

  reset(): void {
    throw new NotImplementedError("Method not implemented.");
  }

  createSnapshot(): unknown {
    throw new NotImplementedError("Method not implemented.");
  }

  loadSnapshot(state: ReturnType<typeof this.createSnapshot>): void {
    throw new NotImplementedError("Method not implemented.");
  }
}
```

_Source: [examples/memory/custom.ts](/typescript/examples/memory/custom.ts)_

The simplest implementation is `UnconstrainedMemory`, which can be found [here](/typescript/src/memory/unconstrainedMemory.ts).
