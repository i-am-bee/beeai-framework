import "dotenv/config";
import { AmazonBedrockChatModel } from "beeai-framework/adapters/amazon-bedrock/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { z } from "zod";
import { ChatModelError } from "beeai-framework/backend/errors";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new AmazonBedrockChatModel(
  "amazon.titan-text-lite-v1",
  // {},
  // {
  //   accessKeyId: "AWS_ACCESS_KEY_ID",
  //   secretAccessKey: "AWS_SECRET_ACCESS_KEY",
  //   region: "AWS_REGION",
  //   sessionToken: "AWS_SESSION_TOKEN",
  // },
);

llm.config({
  parameters: {
    topK: 1,
    temperature: 0,
    topP: 1,
  },
});

async function amazonBedrockFromName() {
  const amazonBedrockLLM = await ChatModel.fromName("amazon-bedrock:amazon.titan-text-lite-v1");
  const response = await amazonBedrockLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function amazonBedrockSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function amazonBedrockStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function amazonBedrockAbort() {
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

async function amazonBedrockStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function amazonBedrockToolCalling() {
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

async function amazonBedrockDebug() {
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

console.info(" amazonBedrockFromName".padStart(25, "*"));
await amazonBedrockFromName();
console.info(" amazonBedrockSync".padStart(25, "*"));
await amazonBedrockSync();
console.info(" amazonBedrockStream".padStart(25, "*"));
await amazonBedrockStream();
console.info(" amazonBedrockAbort".padStart(25, "*"));
await amazonBedrockAbort();
console.info(" amazonBedrockStructure".padStart(25, "*"));
await amazonBedrockStructure();
console.info(" amazonBedrockToolCalling".padStart(25, "*"));
await amazonBedrockToolCalling();
console.info(" amazonBedrockDebug".padStart(25, "*"));
await amazonBedrockDebug();
