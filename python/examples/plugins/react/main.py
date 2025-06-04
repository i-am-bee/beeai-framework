# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using the plugin loader to load and run the plugins defined in the plugins directory."""

import asyncio
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from beeai_framework.agents.experimental.events import RequirementAgentStartEvent, RequirementAgentSuccessEvent
from beeai_framework.backend import ChatModelNewTokenEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.plugins.loader import PluginLoader
from beeai_framework.utils.strings import to_json
from examples.helpers.io import ConsoleReader

load_dotenv()
plugin_name = "GeneralChat"
if len(sys.argv) > 1:
    plugin_name = sys.argv[1]


loader = PluginLoader().root()
loader.load_config("./config.yaml")
plugin = loader.get_plugin(plugin_name)


async def main() -> None:
    """main method."""
    try:
        reader = ConsoleReader()

        for prompt in ["What is the weather in Jackson Wyoming?"]:

            def on_start(data: RequirementAgentStartEvent, meta: EventMeta) -> None:
                reader.write(
                    "Agent 🤖 (start)",
                    to_json(
                        data,
                        indent=2,
                        sort_keys=False,
                    ),
                )

            def on_new_token(data: ChatModelNewTokenEvent, meta: EventMeta) -> None:
                reader.write("LLM 🤖 (new_token)", data.value.get_text_content())

            def on_success(data: RequirementAgentSuccessEvent, meta: EventMeta) -> None:
                reader.write("Requirements 🤖 (success)", data.response.get_text_content())
                if data.response.usage:
                    reader.write("LLM 🤖 (usage)", str(data.response.usage.model_dump()))

            response = (
                await plugin.run(plugin.input_schema(data=prompt))
                .on("start", on_start)
                .on("success", on_success)
                .on("new_token", on_new_token)
            )
            print(response.data.text)
            print("\n")
    except ValidationError as exc:
        print(exc)
        print(repr(exc.errors()[0]["type"]))
        # > 'arguments_type'


asyncio.run(main())
