import "dotenv/config";
import { UserMessage } from "beeai-framework/backend/message";
import { WatsonxChatModel } from "beeai-framework/adapters/watsonx/backend/chat";

const chatLLM = new WatsonxChatModel("ibm/granite-3-3-8b-instruct");

// Log every request
chatLLM.emitter.match("*", async (data, event) => {
  console.info(
    `Time: ${event.createdAt.toISOString().substring(11, 19)}`,
    `Event: ${event.name}`,
    `Data: ${JSON.stringify(data).substring(0, 128).concat("...")}`,
  );
});

const response = await chatLLM.create({
  messages: [new UserMessage("Hello world!")],
});
console.info(response.messages[0]);
