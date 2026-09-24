from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather import OpenMeteoTool

agent = ReActAgent(llm=OllamaChatModel("llama3.1"), tools=[OpenMeteoTool()], memory=UnconstrainedMemory())
