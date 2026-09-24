import { Emitter } from "beeai-framework/emitter/emitter";
import {
  Tool,
  BaseToolOptions,
  BaseToolRunOptions,
  JSONToolOutput,
  ToolInput,
  ToolEmitter,
} from "beeai-framework/tools/base";
import { RunContext } from "beeai-framework/context";
import { z } from "zod";

interface HumanToolOutput {
  clarification: string;
}

export interface Reader {
  write(prefix: string, message: string): void;
  askSingleQuestion(prompt: string, options?: { signal?: AbortSignal }): Promise<string>;
}

export interface HumanToolInput extends BaseToolOptions {
  reader: Reader;
  name?: string;
  description?: string;
}

export class HumanTool extends Tool<JSONToolOutput<HumanToolOutput>, HumanToolInput> {
  name = "HumanTool";
  description = `
  This tool is used whenever the user's input is unclear, ambiguous, or incomplete. 
  The agent MUST invoke this tool when additional clarification is required to proceed. 
  The output must adhere strictly to the following structure:
    - Thought: A single-line description of the need for clarification.
    - Function Name: HumanTool
    - Function Input: { "message": "Your question to the user for clarification." }
    - Function Output: The user's response in JSON format.
  Examples:
    - Example 1:
      Input: "What is the weather?"
      Thought: "The user's request lacks a location. I need to ask for clarification."
      Function Name: HumanTool
      Function Input: { "message": "Could you provide the location for which you would like to know the weather?" }
      Function Output: { "clarification": "Santa Fe, Argentina" }
      Final Answer: The current weather in Santa Fe, Argentina is 17.3°C with a relative humidity of 48% and a wind speed of 10.1 km/h.

    - Example 2:
      Input: "Can you help me?"
      Thought: "The user's request is too vague. I need to ask for more details."
      Function Name: HumanTool
      Function Input: { "message": "Could you clarify what kind of help you need?" }
      Function Output: { "clarification": "I need help understanding how to use the project management tool." }
      Final Answer: Sure, I can help you with the project management tool. Let me know which feature you'd like to learn about or if you'd like a general overview.

    - Example 3:
      Input: "Translate this sentence."
      Thought: "The user's request is incomplete. I need to ask for the sentence they want translated."
      Function Name: HumanTool
      Function Input: { "message": "Could you specify the sentence you would like me to translate?" }
      Function Output: { "clarification": "Translate 'Hello, how are you?' to French." }
      Final Answer: The French translation of 'Hello, how are you?' is 'Bonjour, comment vas-tu?'

  Note: Do NOT attempt to guess or provide incomplete responses. Always use this tool when in doubt to ensure accurate and meaningful interactions.
`;

  public readonly emitter: ToolEmitter<ToolInput<this>, JSONToolOutput<HumanToolOutput>> =
    Emitter.root.child({
      namespace: ["tool", "human"],
      creator: this,
    });

  constructor(protected readonly input: HumanToolInput) {
    super(input);
    this.name = input?.name || this.name;
    this.description = input?.description || this.description;
  }

  inputSchema() {
    return z.object({
      message: z.string().min(1, "Message cannot be empty"),
    });
  }

  async _run(
    input: ToolInput<this>,
    _options: Partial<BaseToolRunOptions>,
    run: RunContext<this>,
  ): Promise<JSONToolOutput<HumanToolOutput>> {
    // Use the reader from input
    this.input.reader.write("HumanTool", input.message);

    // Use askSingleQuestion with the signal
    const userInput = await this.input.reader.askSingleQuestion("User 👤 : ", {
      signal: run.signal,
    });

    // Return JSONToolOutput with the clarification
    return new JSONToolOutput({
      clarification: userInput.trim(),
    });
  }
}
