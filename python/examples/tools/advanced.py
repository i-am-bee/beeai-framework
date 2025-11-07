import asyncio
import datetime
import sys
import traceback

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(
            location_name="New York",
            start_date=datetime.date.today(),
            end_date=datetime.date.today(),
            temperature_unit="celsius",
        )
    )
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
