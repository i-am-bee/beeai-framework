import 'dotenv/config'
import { UserMessage } from "beeai-framework/backend/core";
import { OpenAIChatModel } from "beeai-framework/adapters/openai/backend/chat";
import { z } from "zod";

const model = new OpenAIChatModel('gpt-4.1-nano');
const { object } = await model.createStructure({
  schema: z.object({answer: z.string()}),
  messages: [new UserMessage("What has keys but can’t open locks?")],
  maxRetries: 3,
});
console.log(`Answer: ${object.answer}`);
