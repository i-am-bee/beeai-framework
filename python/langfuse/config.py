from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.beeai import BeeAIInstrumentor

# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()
assert langfuse.auth_check(), "Langfuse auth failed - check your keys ✋"

# Instrument BeeAI
BeeAIInstrumentor().instrument()