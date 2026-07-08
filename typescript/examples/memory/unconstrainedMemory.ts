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
