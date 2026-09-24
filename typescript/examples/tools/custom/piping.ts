import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { SimilarityTool } from "beeai-framework/tools/similarity";
import { splitString } from "beeai-framework/internals/helpers/string";
import { z } from "zod";

const wikipedia = new WikipediaTool();
const similarity = new SimilarityTool({
  maxResults: 5,
  provider: async (input) =>
    input.documents.map((document) => ({
      score: document.text
        .toLowerCase()
        .split(" ")
        .reduce((acc, word) => acc + (input.query.toLowerCase().includes(word) ? 1 : 0), 0),
    })),
});

const wikipediaWithSimilarity = wikipedia
  .extend(
    z.object({
      page: z.string().describe("Wikipedia page"),
      query: z.string().describe("Search query"),
    }),
    (newInput) => ({ query: newInput.page }),
  )
  .pipe(similarity, (input, output) => ({
    query: input.query,
    documents: output.results.flatMap((document) =>
      Array.from(splitString(document.fields.markdown ?? "", { size: 1000, overlap: 50 })).map(
        (chunk) => ({
          text: chunk,
          source: document,
        }),
      ),
    ),
  }));

const response = await wikipediaWithSimilarity.run({
  page: "JavaScript",
  query: "engine",
});
console.info(response);
