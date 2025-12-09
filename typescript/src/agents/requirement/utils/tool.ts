import { AssistantMessage, MessageToolCallContent } from "@/backend";
import { ChatModelToolCallError } from "@/backend/errors";
import { RunContext } from "@/context";
import { Emitter } from "@/emitter";
import { FrameworkError } from "@/errors";
import { AnyTool, StringToolOutput, Tool, ToolError, ToolOutput, ToolRunOptions } from "@/tools";
import { RequirementAgentRunState } from "@/agents/requirement";
import { toJson } from "@/utils/strings";

export interface ToolInvocationResult {
  msg: MessageToolCallContent;
  tool: AnyTool | null;
  input: any;
  output: ToolOutput;
  error: FrameworkError | null;
}

async function _runTool(
  tools: AnyTool[],
  msg: MessageToolCallContent,
  context: Record<string, any>,
): Promise<ToolInvocationResult> {
  if (!msg.isValid()) {
    throw new ChatModelToolCallError(
      toJson({ name: msg.toolName, parameters: msg.args }, { sortKeys: false }),
      "The generated tool call is invalid. Cannot parse the args.",
    );
  }

  const result: ToolInvocationResult = {
    msg,
    tool: null,
    input: JSON.parse(msg.args),
    output: new StringToolOutput(""),
    error: null,
  };

  try {
    result.tool = tools.find((ability) => ability.name === msg.toolName) || null;
    if (!result.tool) {
      throw new ToolError(`Tool '${msg.toolName}' does not exist!`);
    }

    result.output = await result.tool.run(result.input).context({ ...context, toolCallMsg: msg });
  } catch (e) {
    const error = FrameworkError.ensure(e as Error);
    result.error = error;
  }

  return result;
}

async function _runTools(
  tools: AnyTool[],
  messages: MessageToolCallContent[],
  context: Record<string, any>,
): Promise<ToolInvocationResult[]> {
  return await Promise.all(messages.map((msg) => _runTool(tools, msg, context)));
}

export interface FinalAnswerToolSchema {
  response: string; // The final answer to the user
}

export class FinalAnswerTool extends Tool<Record<string, any>, ToolRunOptions, StringToolOutput> {
  name = "final_answer";
  description = "Sends the final answer to the user";
  instructions?: string;
  customSchema: boolean;

  private _expectedOutput: string | (new () => any) | null;
  private _state: RequirementAgentRunState;

  constructor(expectedOutput: string | (new () => any) | null, state: RequirementAgentRunState) {
    super();
    this._expectedOutput = expectedOutput;
    this._state = state;
    this.instructions = typeof expectedOutput === "string" ? expectedOutput : undefined;
    this.customSchema = typeof expectedOutput === "function";
  }

  protected _createEmitter(): Emitter {
    return Emitter.root().child({
      namespace: ["tool", "final_answer"],
      creator: this,
    });
  }

  get inputSchema(): new () => any {
    const expectedOutput = this._expectedOutput;

    if (expectedOutput === null) {
      return class implements FinalAnswerToolSchema {
        response = "";
      };
    } else if (typeof expectedOutput === "function") {
      return expectedOutput;
    } else if (typeof expectedOutput === "string") {
      return class implements FinalAnswerToolSchema {
        response = ""; // expectedOutput as description
      };
    } else {
      return class implements FinalAnswerToolSchema {
        response = "";
      };
    }
  }

  protected async _run(
    input: Record<string, any>,
    options: ToolRunOptions | null,
    context: RunContext,
  ): Promise<StringToolOutput> {
    this._state.result = input;
    if (this.inputSchema === this._expectedOutput) {
      this._state.answer = new AssistantMessage(JSON.stringify(input));
    } else {
      this._state.answer = new AssistantMessage((input as FinalAnswerToolSchema).response);
    }

    return new StringToolOutput("Message has been sent");
  }

  async clone(): Promise<this> {
    const tool = new (this.constructor as new (
      expectedOutput: string | (new () => any) | null,
      state: RequirementAgentRunState,
    ) => this)(this._expectedOutput, { ...this._state });
    tool.name = this.name;
    tool.description = this.description;
    tool._cache = await this.cache.clone();
    tool.middlewares.push(...this.middlewares);
    return tool;
  }
}
