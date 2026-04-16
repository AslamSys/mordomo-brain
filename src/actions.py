"""
Action extraction from LLM response tool_calls.

With function calling, the LLM returns structured tool_calls directly in the
response — no regex, no parsing, no brittle text manipulation.
The extraction is done inside llm.py (_extract_tool_calls); this module
exists as a thin compatibility layer for handlers.py.

The old [ACTION:{...}] regex approach has been removed entirely.
"""
from src.llm import LLMResponse


def extract_actions(llm_response: LLMResponse) -> tuple[str, list[dict]]:
    """
    Returns (response_text, actions) from an LLMResponse.
    Actions come directly from tool_calls — already structured, no parsing needed.
    """
    return llm_response.text, llm_response.tool_calls
