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
