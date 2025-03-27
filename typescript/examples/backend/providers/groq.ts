import "dotenv/config";
import { GroqChatModel } from "beeai-framework/adapters/groq/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { AbortError } from "beeai-framework/errors";
import { z } from "zod";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new GroqChatModel(
  "gemma2-9b-it",
  // {},
  // {
  //   apiKey: "GROQ_API_KEY",
  //   baseURL: "GROQ_API_BASE_URL",
  // },
);

llm.config({
  parameters: {
    temperature: 0.7,
    maxTokens: 1024,
    topP: 1,
  },
});

async function groqFromName() {
  const groqLLM = await ChatModel.fromName("groq:gemma2-9b-it");
  const response = await groqLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function groqSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function groqStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function groqAbort() {
  try {
    const response = await llm.create({
      messages: [new UserMessage("What is the smallest of the Cape Verde islands?")],
      stream: true,
      abortSignal: AbortSignal.timeout(5 * 1000),
    });
    console.info(response.getTextContent());
  } catch (err) {
    if (err instanceof AbortError) {
      console.error("Aborted", { err });
    }
  }
}

async function groqStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function groqToolCalling() {
  const userMessage = new UserMessage("What is the weather in Boston?");
  const weatherTool = new OpenMeteoTool({ retryOptions: { maxRetries: 3 } });
  const response = await llm.create({ messages: [userMessage], tools: [weatherTool] });
  const toolCallMsg = response.getToolCalls()[0];
  console.debug(JSON.stringify(toolCallMsg));
  const toolResponse = await weatherTool.run(toolCallMsg.args as any);
  const toolResponseMsg = new ToolMessage({
    type: "tool-result",
    result: toolResponse.getTextContent(),
    toolName: toolCallMsg.toolName,
    toolCallId: toolCallMsg.toolCallId,
  });
  console.info(toolResponseMsg.toPlain());
  const finalResponse = await llm.create({ messages: [userMessage, toolResponseMsg], tools: [] });
  console.info(finalResponse.getTextContent());
}

async function groqDebug() {
  // Log every request
  llm.emitter.match("*", (value, event) =>
    console.debug(
      `Time: ${event.createdAt.toISOString()}`,
      `Event: ${event.name}`,
      `Data: ${value}`,
    ),
  );

  const response = await llm.create({
    messages: [new UserMessage("Hello world!")],
  });
  console.info(response.messages[0].toPlain());
}

console.info("groqFromName".padStart(25, "*"));
await groqFromName();
console.info("groqSync".padStart(25, "*"));
await groqSync();
console.info("groqStream".padStart(25, "*"));
await groqStream();
console.info("groqAbort".padStart(25, "*"));
await groqAbort();
console.info("groqStructure".padStart(25, "*"));
await groqStructure();
console.info("groqToolCalling".padStart(25, "*"));
await groqToolCalling();
console.info("groqDebug".padStart(25, "*"));
await groqDebug();
