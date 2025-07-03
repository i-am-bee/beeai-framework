import asyncio
import sys
import traceback

from beeai_framework.adapters.watsonx_orchestrate.agents import WatsonxOrchestrateAgent
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = WatsonxOrchestrateAgent(
        # To find your instance URL, visit IBM watsonx Orchestrate -> Settings -> API Details
        # Example: https://api.eu-de.watson-orchestrate.cloud.ibm.com/instances/aaaaaa-bbbb-cccc-dddd-eeeeeeeee
        instance_url="YOUR_INSTANCE_URL",
        # To find agent's ID, visit IBM watsonx Orchestrate -> Select any existing agent -> copy the last part of the URL ()
        # Example: 1xfa8c27-6d0f-4962-9eb5-4e1c0b8073d8
        agent_id="YOUR_AGENT_ID",
        # Auth type, typically IAM (hosted version) or JWT for custom deployments
        auth_type="iam",
        # To find your API Key, visit IBM watsonx Orchestrate -> Settings -> API Details -> Generate API Key
        api_key="YOUR_API_KEY",
    )
    for prompt in reader:
        response = await agent.run(prompt)
        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
