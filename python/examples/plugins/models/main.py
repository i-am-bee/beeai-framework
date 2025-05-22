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

from beeai_framework.backend import ChatModelNewTokenEvent, ChatModelStartEvent, ChatModelSuccessEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.plugins.loader import PluginLoader
from examples.helpers.io import ConsoleReader

load_dotenv()
plugin_name = "SigmaComposer"
if len(sys.argv) > 1:
    plugin_name = sys.argv[1]

PluginLoader.root().load_config("python/examples/plugins/models/config.yaml")
plugin = PluginLoader.root().get_plugin(plugin_name)


async def main() -> None:
    """main method."""
    try:
        reader = ConsoleReader()

        for prompt in reader:

            def on_start(data: ChatModelStartEvent, meta: EventMeta) -> None:
                reader.write("LLM 🤖 (start)", str(data.input.model_dump()))

            def on_new_token(data: ChatModelNewTokenEvent, meta: EventMeta) -> None:
                reader.write("LLM 🤖 (new_token)", data.value.get_text_content())

            def on_success(data: ChatModelSuccessEvent, meta: EventMeta) -> None:
                reader.write("LLM 🤖 (success)", data.value.get_text_content())
                if data.value.usage:
                    reader.write("LLM 🤖 (usage)", str(data.value.usage.model_dump()))

            response = (
                await plugin.run(plugin.input_schema(data=prompt))
                .on("start", on_start)
                .on("success", on_success)
                .on("new_token", on_new_token)
            )

            for message in response.data:
                print(message.text, end="|", flush=True)
            print("\n")
    except ValidationError as exc:
        print(exc)
        print(repr(exc.errors()[0]["type"]))
        # > 'arguments_type'


asyncio.run(main())
