import asyncio
from typing import TypedDict

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from beeai_framework.backend import ChatModel, MessageToolResultContent, UserMessage
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool


# -------------------------
# USAGE METADATA STRUCTURE
# -------------------------
class UsageMetadata(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


# -------------------------
# TOOLS
# -------------------------
@traceable(run_type="tool", name="Wikipedia")
async def wiki_lookup(query: str) -> MessageToolResultContent:
    wiki = WikipediaTool()
    resp = await wiki.run({"query": query})

    rt = get_current_run_tree()
    meta = rt.extra.setdefault("metadata", {})
    meta.update({"query": query, "result": resp})
    return resp


@traceable(run_type="tool", name="OpenMeteo")
async def meteo_lookup(location: str) -> MessageToolResultContent:
    meteo = OpenMeteoTool()
    resp = await meteo.run({"location_name": location})

    rt = get_current_run_tree()
    meta = rt.extra.setdefault("metadata", {})
    meta.update({"location": location, "result": resp})

    return resp


# -------------------------
# LLM HELPERS
# -------------------------
async def _llm_call(prompt: str, step_name: str) -> str:
    llm = ChatModel.from_name("ollama:llama3.1")
    resp = await llm.create(messages=[UserMessage(prompt)])
    output_text = resp.messages[-1].content[0].text

    usage: UsageMetadata = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }

    rt = get_current_run_tree()
    meta = rt.extra.setdefault("metadata", {})
    meta.update({"input": prompt, "output": output_text, "usage": usage})

    return output_text


@traceable(run_type="llm", name="Researcher LLM Call")
async def researcher_llm(prompt: str) -> str:
    return await _llm_call(prompt, "Researcher LLM Call")


@traceable(run_type="llm", name="Weather Forecaster LLM Call")
async def weather_llm(prompt: str) -> str:
    return await _llm_call(prompt, "Weather Forecaster LLM Call")


@traceable(run_type="llm", name="Data Synthesizer LLM Call")
async def synth_llm(prompt: str) -> str:
    return await _llm_call(prompt, "Data Synthesizer LLM Call")


# -------------------------
# MAIN AGENT
# -------------------------
@traceable(run_type="chain", name="BeeAI Agent Workflow")
async def agent(topic: str, location: str) -> str:
    # Step 1: Wikipedia Research
    wiki_data = await wiki_lookup(topic)
    research_summary = await researcher_llm(f"Summarize this for a non-expert: {wiki_data}")

    # Step 2: Weather Forecast
    meteo_data = await meteo_lookup(location)
    weather_summary = await weather_llm(f"Summarize this weather data: {meteo_data}")

    # Step 3: Synthesis
    final_report = await synth_llm(
        f"Combine this research:\n{research_summary}\n"
        f"With this weather info:\n{weather_summary}\n"
        f"into one coherent summary."
    )

    # Add final output to the workflow-level metadata
    rt = get_current_run_tree()
    meta = rt.extra.setdefault("metadata", {})
    meta.update({"topic": topic, "location": location, "final_output": final_report})

    return final_report


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    result = asyncio.run(agent("Climate change", "New York"))
    print("\n===== FINAL REPORT =====\n", result)
