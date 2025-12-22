import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { Requirement, Rule } from "beeai-framework/agents/requirement/requirements/requirement";
import { ConditionalRequirement } from "beeai-framework/agents/requirement/requirements/conditional";
import { ChatModel } from "beeai-framework/backend/chat";
import { RunContext } from "beeai-framework/context";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { AnyTool, Tool } from "beeai-framework/tools/base";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { ThinkTool } from "beeai-framework/tools/think";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { RequirementAgentRunState } from "beeai-framework/agents/requirement/types";
import { GlobalTrajectoryMiddleware } from "beeai-framework/middleware/trajectory";
import {
  RequirementAgentSystemPrompt,
  RequirementAgentTaskPrompt,
  RequirementAgentToolErrorPrompt,
  RequirementAgentToolNoResultPrompt,
} from "beeai-framework/agents/requirement/prompts";

class RepeatIfEmptyRequirement extends Requirement {
  /** Custom requirement that repeats the last tool if its output is empty. */

  protected targetCls: new (...args: any[]) => AnyTool;
  protected targets: AnyTool[] = [];
  protected limit: number;
  protected remaining: number;

  constructor(target: new (...args: any[]) => AnyTool, options: { limit?: number } = {}) {
    super(`RepeatIfEmpty${target.name}`);
    this.priority = 20;
    this.targetCls = target;
    this.limit = options.limit ?? Infinity;
    this.remaining = this.limit;
  }

  // this part is optional (you don't need to verify whether tools exist)
  async init(tools: AnyTool[], ctx: RunContext<any>): Promise<void> {
    await super.init(tools, ctx);

    this.targets = tools.filter((tool) => tool instanceof this.targetCls);
    if (this.targets.length === 0) {
      throw new Error(`No tool of type ${this.targetCls.name} found!`);
    }
  }

  async _run(state: RequirementAgentRunState, _: RunContext<typeof this>): Promise<Rule[]> {
    const lastStep = state.steps?.at(-1);

    if (lastStep && this.targets.includes(lastStep.tool!) && lastStep.output.isEmpty()) {
      this.remaining--;
      return [
        {
          target: lastStep.tool!.name,
          allowed: true,
          forced: true,
          hidden: false,
          preventStop: false,
        },
      ];
    } else {
      this.remaining = this.limit;
      return [];
    }
  }
}

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:granite4:micro"),
  tools: [new ThinkTool(), new WikipediaTool(), new OpenMeteoTool()],
  role: "a trip planner",
  instructions: [
    "Plan activities for a given destination based on current weather and events.",
    "Input to Wikipedia should be a name of the target city.",
  ],
  name: "PlannerAgent", // optional, useful for Handoff or registering agent to AgentStack or others
  description: "Assistant to plan your day in a given destination.", // optional
  requirements: [
    new ConditionalRequirement(ThinkTool, {
      forceAtStep: 1,
      forceAfter: [Tool],
      consecutiveAllowed: false,
    }), // ReAct
    new ConditionalRequirement(OpenMeteoTool, { onlyAfter: [WikipediaTool] }),
    new ConditionalRequirement(WikipediaTool, { maxInvocations: 5 }),
    new RepeatIfEmptyRequirement(WikipediaTool, { limit: 3 }),
  ],
  saveIntermediateSteps: true, // store tool calls between individual starts (default: true)
  toolCallChecker: true, // detects and resolve cycles (default: true)
  finalAnswerAsTool: true, // produces the final answer as a tool call (default: true)
  memory: new UnconstrainedMemory(),
  templates: {
    system: RequirementAgentSystemPrompt,
    task: RequirementAgentTaskPrompt,
    toolError: RequirementAgentToolErrorPrompt,
    toolNoResult: RequirementAgentToolNoResultPrompt,
  },
  middlewares: [
    new GlobalTrajectoryMiddleware({
      included: [Tool],
    }),
  ],
});

const response = await agent.run({
  prompt: "What to do in Boston?",
  context: "I already visited Freedom Trail.",
  // one can pass a Zod schema to get a structured output
  expectedOutput:
    "Detailed plan on what to do from morning to evening, split in sections each with a time range.",
});

console.log(response.result.text);
// console.log(response.memory);  // temp memory created
// console.log(response.state.iteration);  // number of iterations (steps)
// console.log(response.state.steps);  // individual steps
// for (const step of response.state.steps) {
//   console.log("Iteration", step.iteration);
//   if (step.tool) {
//     console.log("-> Tool", step.tool.name);
//   }
//   console.log("-> Input", step.input);
//   console.log("-> Output", step.output);
//   console.log("-> Error", step.error);
// }
