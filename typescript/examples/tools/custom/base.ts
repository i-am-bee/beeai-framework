import {
  ToolEmitter,
  StringToolOutput,
  Tool,
  ToolInput,
  ToolInputValidationError,
} from "beeai-framework/tools/base";
import { z } from "zod";
import { randomInteger } from "remeda";
import { Emitter } from "beeai-framework/emitter/emitter";

export class RiddleTool extends Tool<StringToolOutput> {
  name = "Riddle";
  description = "It generates a random puzzle to test your knowledge.";

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "riddle"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      index: z
        .number()
        .int()
        .min(0)
        .max(RiddleTool.data.length - 1)
        .optional(),
    });
  }

  public static data = [
    "What has hands but can’t clap?",
    "What has a face and two hands but no arms or legs?",
    "What gets wetter the more it dries?",
    "What has to be broken before you can use it?",
    "What has a head, a tail, but no body?",
    "The more you take, the more you leave behind. What am I?",
    "What goes up but never comes down?",
  ];

  static {
    // Makes the class serializable
    this.register();
  }

  protected async _run(input: ToolInput<this>): Promise<StringToolOutput> {
    const index = input.index ?? randomInteger(0, RiddleTool.data.length - 1);
    const riddle = RiddleTool.data[index];
    if (!riddle) {
      throw new ToolInputValidationError(`Riddle with such index (${index}) does not exist!`);
    }
    return new StringToolOutput(riddle);
  }
}
