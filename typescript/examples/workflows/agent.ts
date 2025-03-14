import "dotenv/config";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { z } from "zod";
import { Message, UserMessage } from "beeai-framework/backend/message";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { ReadOnlyMemory } from "beeai-framework/memory/base";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Workflow } from "beeai-framework/workflows/workflow";
import { createConsoleReader } from "examples/helpers/io.js";
import { GroqChatModel } from "beeai-framework/adapters/groq/backend/chat";

const schema = z.object({
  answer: z.instanceof(Message).optional(),
  memory: z.instanceof(ReadOnlyMemory),
});

const workflow = new Workflow({ schema: schema })
  .addStep("simpleAgent", async (state) => {
    const simpleAgent = new ReActAgent({
      llm: new GroqChatModel("llama-3.3-70b-versatile"),
      tools: [],
      memory: state.memory,
    });
    const answer = await simpleAgent.run({ prompt: null });
    reader.write("🤖 Simple Agent", answer.result.text);

    state.answer = answer.result;
    return "critique";
  })
  .addStrictStep("critique", schema.required(), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const { object: critiqueResponse } = await llm.createStructure({
      schema: z.object({ score: z.number().int().min(0).max(100) }),
      messages: [
        Message.of({
          role: "system",
          text: `You are an evaluation assistant who scores the credibility of the last assistant's response. Chitchatting always has a score of 100. If the assistant was unable to answer the user's query, then the score will be 0.`,
        }),
        ...state.memory.messages,
        state.answer,
      ],
    });
    reader.write("🧠 Score", critiqueResponse.score.toString());

    return critiqueResponse.score < 75 ? "complexAgent" : Workflow.END;
  })
  .addStep("complexAgent", async (state) => {
    const complexAgent = new ReActAgent({
      llm: new GroqChatModel("llama-3.3-70b-versatile"),
      tools: [new WikipediaTool(), new OpenMeteoTool()],
      memory: state.memory,
    });
    const { result } = await complexAgent.run({ prompt: null });
    reader.write("🤖 Complex Agent", result.text);
    state.answer = result;
  })
  .setStart("simpleAgent");

const reader = createConsoleReader();
const memory = new UnconstrainedMemory();

for await (const { prompt } of reader) {
  const userMessage = new UserMessage(prompt);
  await memory.add(userMessage);

  const response = await workflow.run({
    memory: memory.asReadOnly(),
  });
  await memory.add(response.state.answer!);

  reader.write("🤖 Final Answer", response.state.answer!.text);
}
