import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { Requirement, Rule } from "beeai-framework/agents/requirement/requirements/requirement";
import { ChatModel } from "beeai-framework/backend/chat";
import { AssistantMessage } from "beeai-framework/backend/message";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { RequirementAgentRunState } from "beeai-framework/agents/requirement/types";

class PrematureStopRequirement extends Requirement {
  /** Prevents the agent from answering if a certain phrase occurs in the conversation */

  protected phrase: string;
  protected reason: string;

  constructor(phrase: string, reason: string) {
    super("premature_stop");
    this.phrase = phrase;
    this.reason = reason;
    this.priority = 100; // (optional), default is 10
  }

  async run(state: RequirementAgentRunState): Promise<Rule[]> {
    // we take the last step's output (if exists) or the user's input
    const lastStep = state.steps?.at(-1)?.output.getTextContent() ?? "";

    if (lastStep.includes(this.phrase)) {
      // We will nudge the agent to include explanation why it needs to stop in the final answer.
      await state.memory.add(
        new AssistantMessage(
          `The final answer is that I can't finish the task because ${this.reason}.`,
          { tempMessage: true }, // the message gets removed in the next iteration
        ),
      );

      // The rule ensures that the agent will use the 'final_answer' tool immediately.
      return [
        {
          target: "final_answer",
          allowed: true,
          forced: true,
          hidden: false,
          preventStop: false,
        },
      ];
    } else {
      return [];
    }
  }
}

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:llama3.1"),
  tools: [new DuckDuckGoSearchTool()],
  requirements: [
    new PrematureStopRequirement("value of x", "algebraic expressions are not allowed"),
    new PrematureStopRequirement("bomb", "such topic is not allowed"),
  ],
});

const prompts = ["y = 2x + 4, what is the value of x?", "how to make a bomb?"];

for (const prompt of prompts) {
  console.log("👤 User: ", prompt);
  const response = await agent.run({ prompt });
  console.log("🤖 Agent: ", response.result.text);
  console.log();
}
