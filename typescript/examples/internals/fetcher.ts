/**
 * Example usage of RestfulClient for consuming ACP server
 */

import { RestfulClient } from "beeai-framework/internals/fetcher";

const client = new RestfulClient({
  baseUrl: "http://127.0.0.1:8000",
  paths: {
    runs: "/runs",
  },
});

const generator = client.stream("runs", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    agent_name: "chat_agent",
    mode: "stream",
    input: [
      {
        parts: [
          {
            name: "text",
            content_type: "text/plain",
            content: "Hello agent!",
            content_encoding: "plain",
            role: "user",
          },
        ],
      },
    ],
  }),
});

for await (const event of generator) {
  const data = JSON.parse(event.data);
  console.info(data);
}
