import "dotenv/config.js";
import { createConsoleReader } from "examples/helpers/io.js";
import { UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";

const llm = await ChatModel.fromName("ollama:granite3.2:8b");

const reader = createConsoleReader();

for await (const { prompt } of reader) {
  const response = await llm.create({
    messages: [new UserMessage(prompt)],
  });
  reader.write(`LLM 🤖 (txt) : `, response.getTextContent());
  reader.write(`LLM 🤖 (raw) : `, JSON.stringify(response.messages));
}
