from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.bee import BeeAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

agent = BeeAgent(llm=OllamaChatModel("llama3.1"), tools=[OpenMeteoTool()], memory=UnconstrainedMemory())
