import "dotenv/config.js";
import { UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { z } from "zod";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ChatModelError } from "beeai-framework/backend/errors";

const llm = new OllamaChatModel(
  "llama3.1",
  {},
  // {
  //   baseURL: "OLLAMA_BASE_URL",
  // },
);

llm.config({
  parameters: {
    topK: 1,
    temperature: 0,
    topP: 1,
  },
});

async function ollamaFromName() {
  const ollamaLLM = await ChatModel.fromName("ollama:llama3.1");
  const response = await ollamaLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function ollamaSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function ollamaStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function ollamaAbort() {
  try {
    const response = await llm.create({
      messages: [new UserMessage("What is the smallest of the Cape Verde islands?")],
      stream: true,
      abortSignal: AbortSignal.timeout(1 * 1000),
    });
    console.info(response.getTextContent());
  } catch (err) {
    if (err instanceof ChatModelError) {
      console.error("Aborted", { err });
    }
  }
}

async function ollamaStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function ollamaDebug() {
  // Log every request
  llm.emitter.match("*", (value, event) =>
    console.debug(
      `Time: ${event.createdAt.toISOString()}`,
      `Event: ${event.name}`,
      `Data: ${JSON.stringify(value)}`,
    ),
  );

  const response = await llm.create({
    messages: [new UserMessage("Hello world!")],
  });
  console.info(response.messages[0].toPlain());
}

console.info(" ollamaFromName".padStart(25, "*"));
await ollamaFromName();
console.info(" ollamaSync".padStart(25, "*"));
await ollamaSync();
console.info(" ollamaStream".padStart(25, "*"));
await ollamaStream();
console.info(" ollamaAbort".padStart(25, "*"));
await ollamaAbort();
console.info(" ollamaStructure".padStart(25, "*"));
await ollamaStructure();
console.info(" ollamaDebug".padStart(25, "*"));
await ollamaDebug();
