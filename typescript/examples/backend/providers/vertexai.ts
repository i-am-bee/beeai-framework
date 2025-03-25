import "dotenv/config.js";
import { GoogleVertexChatModel } from "beeai-framework/adapters/google-vertex/backend/chat";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { z } from "zod";
import { ChatModelError } from "beeai-framework/backend/errors";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new GoogleVertexChatModel(
  "gemini-1.5-flash-001",
  // {},
  // {
  //   location: "GOOGLE_VERTEX_LOCATION",
  //   project: "GOOGLE_VERTEX_PROJECT",
  //   googleAuthOptions: {
  //     apiKey: "",
  //   },
  // },
);

llm.config({
  parameters: {
    topK: 1,
    temperature: 0,
    topP: 1,
  },
});

async function googleVertexFromName() {
  const googleVertexLLM = await ChatModel.fromName("google-vertex:gemini-1.5-flash-001");
  const response = await googleVertexLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function googleVertexSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function googleVertexStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function googleVertexAbort() {
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

async function googleVertexStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function googleVertexToolCalling() {
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

async function googleVertexDebug() {
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

console.info(" googleVertexFromName".padStart(25, "*"));
await googleVertexFromName();
console.info(" googleVertexSync".padStart(25, "*"));
await googleVertexSync();
console.info(" googleVertexStream".padStart(25, "*"));
await googleVertexStream();
console.info(" googleVertexAbort".padStart(25, "*"));
await googleVertexAbort();
console.info(" googleVertexStructure".padStart(25, "*"));
await googleVertexStructure();
console.info(" googleVertexToolCalling".padStart(25, "*"));
await googleVertexToolCalling();
console.info(" googleVertexDebug".padStart(25, "*"));
await googleVertexDebug();
