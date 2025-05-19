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

import asyncio
import random

from pydantic import BaseModel

from beeai_framework.workflows.experimental.workflow import Workflow, fan_in, listen, router, start


class MyAgentState(BaseModel):
    a: int = 0
    b: int = 0
    c: int = 0
    status: str = ""


class MyAgent(Workflow[MyAgentState]):
    @start
    async def start(self) -> None:
        """Start the agent workflow"""
        print("Starting")
        print("Start state:", self.state)

    @router("start")
    async def route(self) -> str:
        """Conditional routing"""
        print("Routing")
        if random.randint(0, 1) > 0.5:
            self.state.status = "proceed"
            return "proceed"
        else:
            self.state.status = "stop"
            return "stop"

    @listen("stop")
    async def stop_func(self) -> None:
        """Listen for a conditional route"""
        print("Stop event triggered")
        print("Final state:", self.state)

    @listen("proceed")
    async def load_data(self) -> None:
        """Listen for a conditional route"""
        print("Proceed event triggered")

    @listen("load_data")
    async def process_data_a(self) -> None:
        """Static fan out"""
        self.state.a += 1
        await asyncio.sleep(3)
        print("Processed a...")

    @listen("load_data")
    async def process_data_b(self) -> None:
        """Static fan out"""
        self.state.b += 1
        await asyncio.sleep(2)
        print("Processed b...")

    @listen("load_data")
    async def process_data_c(self) -> None:
        """Static fan out"""
        self.state.c += 1
        await asyncio.sleep(1)
        print("Processed c...")

    @listen("process_data_a", "process_data_b", "process_data_c")
    async def do_fan_out(self) -> None:
        """Static fan in, AND logic"""
        print("Processing complete")
        print("State is now:", self.state)
        print("Dynamic fan out")
        # Dynamically fan out
        await self.fan_out("process", [["a"], ["b"], ["c"]], fan_in="fan_in")

    async def process(self, item: str) -> str:
        if item == "a":
            self.state.a += 1
            await asyncio.sleep(3)
        elif item == "b":
            self.state.b += 1
            await asyncio.sleep(2)
        elif item == "c":
            self.state.c += 1
            await asyncio.sleep(1)

        print(f"Fan out function called: {item}")
        return item

    @fan_in("fan_in")
    async def done(self, results: list[str]) -> None:
        """Waits on fan_in event from the fan_out operation"""
        for r in results:
            print(r)

        print("Final state:", self.state)
        print("Done!")


async def main() -> None:
    agent = MyAgent()
    await agent.run(MyAgentState())

    print("########")

    agent2 = MyAgent()
    await agent2.run(MyAgentState())


asyncio.run(main())
