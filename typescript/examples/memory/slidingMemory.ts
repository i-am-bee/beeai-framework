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
