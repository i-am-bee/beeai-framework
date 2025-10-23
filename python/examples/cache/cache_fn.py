import asyncio
import random
import sys
import traceback
from typing import TypedDict

from beeai_framework.cache import CacheFn
from beeai_framework.errors import FrameworkError


class TokenResponse(TypedDict):
    token: str
    expires_in: float


async def main() -> None:
    async def fetch_api_token() -> str:
        response: TokenResponse = {"token": f"TOKEN-{random.randint(1000, 9999)}", "expires_in": 0.2}
        get_token.update_ttl(response["expires_in"])
        await asyncio.sleep(0.05)
        return response["token"]

    get_token = CacheFn.create(fetch_api_token, default_ttl=0.1)

    first = await get_token()
    second = await get_token()
    print(first == second)  # True -> cached value

    await asyncio.sleep(0.25)
    refreshed = await get_token()
    print(first == refreshed)  # False -> TTL elapsed, value refreshed


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
