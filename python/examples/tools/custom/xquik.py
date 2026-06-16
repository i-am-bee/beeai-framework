import asyncio
import json
import os
import sys
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import JSONToolOutput, Tool, ToolError, ToolRunOptions


class XquikSearchTweetsToolInput(BaseModel):
    query: str = Field(description="X search query with standard search operators.")
    query_type: Literal["Latest", "Top"] = Field(default="Latest", description="Sort order.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum tweets to return.")


class XquikSearchTweetsToolOutput(JSONToolOutput[dict[str, Any]]):
    pass


def fetch_xquik_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json", "x-api-key": api_key},
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ToolError(
            "Request to Xquik API failed.",
            cause=RuntimeError(detail),
            context={"status_code": exc.code},
        ) from exc
    except URLError as exc:
        raise ToolError("Could not connect to Xquik API.", cause=exc) from exc

    result = json.loads(body)
    if not isinstance(result, dict):
        raise ToolError("Xquik API returned an unexpected response shape.")

    return result


class XquikSearchTweetsTool(Tool[XquikSearchTweetsToolInput, ToolRunOptions, XquikSearchTweetsToolOutput]):
    name = "XquikSearchTweets"
    description = "Searches X posts through the Xquik REST API."
    input_schema = XquikSearchTweetsToolInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "xquik"],
            creator=self,
        )

    async def _run(
        self,
        input: XquikSearchTweetsToolInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> XquikSearchTweetsToolOutput:
        api_key = os.getenv("XQUIK_API_KEY")
        if not api_key:
            raise ToolError("Set XQUIK_API_KEY before running the Xquik search example.")

        base_url = os.getenv("XQUIK_BASE_URL", "https://xquik.com/api/v1").rstrip("/")
        params = urlencode(
            {
                "q": input.query,
                "queryType": input.query_type,
                "limit": input.limit,
            }
        )
        result = await asyncio.to_thread(fetch_xquik_json, f"{base_url}/x/tweets/search?{params}", api_key)

        return XquikSearchTweetsToolOutput(result)


async def main() -> None:
    if not os.getenv("XQUIK_API_KEY"):
        print("Set XQUIK_API_KEY to run this example.")
        return

    tool = XquikSearchTweetsTool()
    result = await tool.run(
        XquikSearchTweetsToolInput(
            query="from:xquikcom",
            limit=5,
        )
    )
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())
