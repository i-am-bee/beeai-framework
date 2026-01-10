import sys
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# --- PATH FIX: Ensures we can import RagasLLM from the same directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ------------------------------------------------------------------------

# Now we can safely import the class
from RagasLLM import RagasLLM

# LangChain imports
from langchain_core.prompt_values import StringPromptValue
from langchain_core.outputs import LLMResult
import asyncio
# ------------------------------------------------------------------
# Test Case 1: Factory Method
# ------------------------------------------------------------------

def test_ragas_llm_factory_method():
    """
    Tests the 'from_name' static method.
    """
    
    # IMPORTANT: Adjust this string if your internal import changes in RagasLLM.py
    # This mocks 'ChatModel' specifically inside the RagasLLM module scope
    target_to_patch = "RagasLLM.ChatModel"

    with patch(target_to_patch) as MockChatModel:
        # Arrange
        fake_internal_model = MagicMock()
        MockChatModel.from_name.return_value = fake_internal_model

        # Act
        adapter = RagasLLM.from_name("gpt-4-turbo", api_key="secret")

        # Assert
        assert isinstance(adapter, RagasLLM)
        assert adapter.model == fake_internal_model
        MockChatModel.from_name.assert_called_once_with("gpt-4-turbo", api_key="secret")


# ------------------------------------------------------------------
# Test Case 2: Async Generation Flow
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agenerate_flow():
    """
    Tests the 'agenerate_text' async method logic.
    """

    # Arrange: Mock the internal model and response
    mock_internal_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.get_text_content.return_value = "Mocked answer"
    mock_internal_model.run.return_value = mock_response

    # Initialize adapter manually
    adapter = RagasLLM(model=mock_internal_model)

    # Input prompt
    input_prompt = StringPromptValue(text="Hi AI")

    # Act
    result = await adapter.agenerate_text(prompt=input_prompt, temperature=0.1)

    # Assert
    assert isinstance(result, LLMResult)
    assert result.generations[0][0].text == "Mocked answer"
    
    # Verify internal call arguments
    args, kwargs = mock_internal_model.run.call_args
    assert kwargs['temperature'] == 0.1
    
    # Verify message content (generic check)
    sent_messages = args[0]
    assert "Hi AI" in str(sent_messages[0]) or getattr(sent_messages[0], 'content', '') == "Hi AI"
    
    
async def main():
    print("Initializing Adapter...")
    llm = RagasLLM.from_name("vertexai:gemini-2.0-flash-lite-001")

    my_prompt = StringPromptValue(text="Explain quantum physics in one sentence.")

    print(f"Sending prompt: '{my_prompt.to_string()}'")

    result = await llm.agenerate_text(
        prompt=my_prompt,
        temperature=0.5  
    )

    generated_text = result.generations[0][0].text

    print("\n✅ Response from Model:")
    print("-" * 30)
    print(generated_text)
    print("-" * 30)


if __name__ == "__main__":
    asyncio.run(main())