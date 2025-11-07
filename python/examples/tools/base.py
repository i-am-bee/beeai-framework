import asyncio
import datetime
import sys

from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(
            location_name="New York", start_date=datetime.date.today(), end_date=datetime.date.today()
        )
    ).middleware(GlobalTrajectoryMiddleware())
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())
