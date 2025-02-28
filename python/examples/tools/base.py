import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(location_name="New York", start_date="2025-01-01", end_date="2025-01-02")
    )
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
