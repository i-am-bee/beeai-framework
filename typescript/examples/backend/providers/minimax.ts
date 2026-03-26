import "dotenv/config";
import { MiniMaxChatModel } from "beeai-framework/adapters/minimax/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { AbortError } from "beeai-framework/errors";
import { z } from "zod";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new MiniMaxChatModel(
  "MiniMax-M2.7",
  // {},
  // {
  //   apiKey: "MINIMAX_API_KEY",
  //   baseURL: "https://api.minimax.io/v1",
  // },
);

llm.config({
  parameters: {
    temperature: 0.7,
    maxTokens: 1024,
    topP: 1,
  },
});

async function minimaxFromName() {
  const minimaxLLM = await ChatModel.fromName("minimax:MiniMax-M2.7");
  const response = await minimaxLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function minimaxSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function minimaxStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function minimaxAbort() {
  try {
    const response = await llm.create({
      messages: [new UserMessage("What is the smallest of the Cape Verde islands?")],
      stream: true,
      abortSignal: AbortSignal.timeout(5 * 1000),
    });
    console.info(response.getTextContent());
  } catch (err) {
    if (err instanceof AbortError) {
      console.log("Aborted", { err });
    }
  }
}

async function minimaxStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function minimaxToolCalling() {
  const userMessage = new UserMessage(
    `What is the current weather in Boston? Current date is ${new Date().toISOString().split("T")[0]}.`,
  );
  const weatherTool = new OpenMeteoTool({ retryOptions: { maxRetries: 3 } });
  const response = await llm.create({
    messages: [userMessage],
    tools: [weatherTool],
  });
  const toolCallMsg = response.getToolCalls()[0];
  console.debug(JSON.stringify(toolCallMsg));
  const toolResponse = await weatherTool.run(toolCallMsg.input as any);
  const toolResponseMsg = new ToolMessage({
    type: "tool-result",
    output: { type: "text", value: toolResponse.getTextContent() },
    toolName: toolCallMsg.toolName,
    toolCallId: toolCallMsg.toolCallId,
  });
  console.info(toolResponseMsg.toPlain());
  const finalResponse = await llm.create({
    messages: [userMessage, ...response.messages, toolResponseMsg],
    tools: [],
  });
  console.info(finalResponse.getTextContent());
}

async function minimaxDebug() {
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

console.info("minimaxFromName".padStart(25, "*"));
await minimaxFromName();
console.info("minimaxSync".padStart(25, "*"));
await minimaxSync();
console.info("minimaxStream".padStart(25, "*"));
await minimaxStream();
console.info("minimaxAbort".padStart(25, "*"));
await minimaxAbort();
console.info("minimaxStructure".padStart(25, "*"));
await minimaxStructure();
console.info("minimaxToolCalling".padStart(25, "*"));
await minimaxToolCalling();
console.info("minimaxDebug".padStart(25, "*"));
await minimaxDebug();
