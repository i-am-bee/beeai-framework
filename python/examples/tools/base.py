import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = tool.run(input=OpenMeteoToolInput(location_name="New York"))
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
