import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { UserMessage } from "beeai-framework/backend/message";

// String containing serialized `UnconstrainedMemory` instance with one message in it.
const serialized = `{"__version":"0.0.0","__root":{"__serializer":true,"__class":"Object","__ref":"18","__value":{"target":"UnconstrainedMemory","snapshot":{"__serializer":true,"__class":"Object","__ref":"17","__value":{"messages":{"__serializer":true,"__class":"Array","__ref":"1","__value":[{"__serializer":true,"__class":"SystemMessage","__ref":"2","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"3","__value":[{"__serializer":true,"__class":"Object","__ref":"4","__value":{"type":"text","text":"You are a helpful assistant."}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"5","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"6","__value":"2025-02-06T14:51:01.459Z"}}},"role":"system"}},{"__serializer":true,"__class":"UserMessage","__ref":"7","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"8","__value":[{"__serializer":true,"__class":"Object","__ref":"9","__value":{"type":"text","text":"Hello!"}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"10","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"11","__value":"2025-02-06T14:51:01.459Z"}}},"role":"user"}},{"__serializer":true,"__class":"AssistantMessage","__ref":"12","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"13","__value":[{"__serializer":true,"__class":"Object","__ref":"14","__value":{"type":"text","text":"Hello, how can I help you?"}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"15","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"16","__value":"2025-02-06T14:51:01.459Z"}}},"role":"assistant"}}]}}}}}}`;

// If `Message` was not imported the serialization would fail because the `Message` had no chance to register itself.
const memory = await UnconstrainedMemory.fromSerialized(serialized, {
  // this part can be omitted if all classes used in the serialized string are imported (and have `static` register block) or at least one initiated
  extraClasses: [UserMessage],
});
console.info(memory.messages);
