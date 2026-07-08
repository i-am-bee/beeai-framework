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
