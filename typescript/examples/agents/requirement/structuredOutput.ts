import { RequirementAgent } from "beeai-framework/agents/requirement/agent";
import { ChatModel } from "beeai-framework/backend/chat";
import { z } from "zod";

const agent = new RequirementAgent({
  llm: await ChatModel.fromName("ollama:llama3.1"),
});

const CharacterSchema = z.object({
  firstName: z.string(),
  lastName: z.string(),
  age: z.number().int(),
  bio: z.string(),
  country: z.string(),
});

const CharactersSchema = z
  .array(CharacterSchema)
  .min(5)
  .max(5)
  .describe("A list of fictional characters");

const response = await agent.run({
  prompt: "Generate fictional characters",
  expectedOutput: CharactersSchema,
});

const characters = response.state.result;

if (Array.isArray(characters)) {
  characters.forEach((character, index) => {
    console.log("Index:", index);
    console.log("-> Full Name:", character.firstName, character.lastName);
    console.log("-> Age:", character.age);
    console.log("-> Country:", character.country);
    console.log("-> Bio:", character.bio);
    console.log();
  });
}
