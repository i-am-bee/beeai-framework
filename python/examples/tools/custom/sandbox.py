import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.code.sandbox import SandboxTool

load_dotenv()


async def main() -> None:
    sandbox_tool = await SandboxTool.from_source_code(
        url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        env={"API_URL": "https://riddles-api.vercel.app/random"},
        source_code="""
import requests
import os
from typing import Optional, Union, Dict

def get_riddle() -> Optional[Dict[str, str]]:
    '''
    Fetches a random riddle from the Riddles API.

    This function retrieves a random riddle and its answer. It does not accept any input parameters.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing:
            - 'riddle' (str): The riddle question.
            - 'answer' (str): The answer to the riddle.
        Returns None if the request fails.
    '''
    url = os.environ.get('API_URL')

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None
""",
    )

    result = await sandbox_tool.run({})

    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
