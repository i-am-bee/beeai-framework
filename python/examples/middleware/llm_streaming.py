import asyncio

from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.emitter import EventMeta
from beeai_framework.middleware.stream_tool_call import StreamToolCallMiddleware, StreamToolCallMiddlewareUpdateEvent
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")
    weather_tool = OpenMeteoTool()
    middleware = StreamToolCallMiddleware(
        weather_tool,
        key="location_name",  # name is taken from the OpenMeteoToolInput schema
        match_nested=False,  # we are applying the middleware to the model directly
        force_streaming=True,  # we want to let middleware enable streaming on the model
    )

    @middleware.emitter.on("update")
    def log_thoughts(event: StreamToolCallMiddlewareUpdateEvent, meta: EventMeta) -> None:
        print(
            "Received update", event.delta, event.output_structured
        )  # event.delta contains an update of the 'location_name' field

    response = await llm.run([UserMessage("What's the current weather in New York?")], tools=[weather_tool]).middleware(
        middleware
    )
    print(response.get_tool_calls()[0].args)


if __name__ == "__main__":
    asyncio.run(main())
