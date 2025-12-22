import "dotenv/config";
import { LLMTool } from "beeai-framework/tools/llm";
import { Tool } from "beeai-framework/tools/base";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Message } from "beeai-framework/backend/message";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const memory = new UnconstrainedMemory();
await memory.addMany([
  Message.of({ role: "system", text: "You are a helpful assistant." }),
  Message.of({ role: "user", text: "Hello!" }),
  Message.of({ role: "assistant", text: "Hello user. I am here to help you." }),
]);

const tool = new LLMTool({
  llm: new OllamaChatModel("granite4:micro"),
});

const response = await tool
  .run({
    task: "Classify whether the tone of text is POSITIVE/NEGATIVE/NEUTRAL.",
  })
  .context({
    // if the context is not passed, the tool will throw an error
    [Tool.contextKeys.Memory]: memory,
  });

console.info(response.getTextContent());
