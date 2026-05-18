import { DynamicTool, JSONToolOutput, ToolError } from "beeai-framework/tools/base";
import { z } from "zod";

type XquikSearchTweetsResponse = Record<string, unknown>;

const getBaseUrl = () =>
  (process.env.XQUIK_BASE_URL ?? "https://xquik.com/api/v1").replace(/\/$/, "");

export const xquikSearchTweetsTool = new DynamicTool({
  name: "XquikSearchTweets",
  description: "Searches X posts through the Xquik REST API.",
  inputSchema: z.object({
    query: z.string().min(1).describe("X search query with standard search operators."),
    queryType: z.enum(["Latest", "Top"]).default("Latest").describe("Sort order."),
    limit: z.number().int().min(1).max(20).default(5).describe("Maximum tweets to return."),
  }),
  async handler(input, _options, run) {
    const apiKey = process.env.XQUIK_API_KEY;
    if (!apiKey) {
      throw new ToolError("Set XQUIK_API_KEY before running the Xquik search example.");
    }

    const url = new URL(`${getBaseUrl()}/x/tweets/search`);
    url.searchParams.set("q", input.query);
    url.searchParams.set("queryType", input.queryType);
    url.searchParams.set("limit", input.limit.toString());

    const response = await fetch(url, {
      headers: {
        "Accept": "application/json",
        "x-api-key": apiKey,
      },
      signal: run.signal,
    });

    if (!response.ok) {
      throw new ToolError("Request to Xquik API failed.", [new Error(await response.text())], {
        context: { statusCode: response.status },
      });
    }

    const result = (await response.json()) as XquikSearchTweetsResponse;
    return new JSONToolOutput(result);
  },
});

if (!process.env.XQUIK_API_KEY) {
  console.log("Set XQUIK_API_KEY to run this example.");
} else {
  const result = await xquikSearchTweetsTool.run({
    query: "from:xquikcom",
    limit: 5,
  });
  console.log(result.getTextContent());
}
