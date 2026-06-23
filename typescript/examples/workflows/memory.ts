import "dotenv/config.js";
import { createConsoleReader } from "examples/helpers/io.js";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Workflow } from "beeai-framework/workflows/workflow";
import { z } from "zod";

// State with memory and output
const schema = z.object({
  memory: z.instanceof(UnconstrainedMemory),
  output: z.string().default(""),
});

type State = z.infer<typeof schema>;

// Echo step: reverse the last message and store as output
async function echo(state: State): Promise<typeof Workflow.END> {
  const lastMessage = state.memory.messages.at(-1);
  state.output = lastMessage ? [...lastMessage.text].reverse().join("") : "";
  return Workflow.END;
}

const workflow = new Workflow({ schema }).addStep("echo", echo);
const memory = new UnconstrainedMemory();

const reader = createConsoleReader();
for await (const { prompt } of reader) {
  // Add user message to memory
  await memory.add(new UserMessage(prompt));
  // Run workflow with shared memory instance
  const response = await workflow.run({ memory, output: "" });
  // Add assistant response to memory
  await memory.add(new AssistantMessage(response.result.output));

  reader.write("Assistant 🤖 : ", response.result.output);
}
