"""
Action extraction from LLM response tool_calls.

With function calling, the LLM returns structured tool_calls directly in the
response — no regex, no parsing, no brittle text manipulation.
"""
from src.llm import LLMResponse


def extract_actions(llm_response: LLMResponse) -> tuple[str, list[dict]]:
    """
    Returns (response_text, actions) from an LLMResponse.
    Actions come directly from tool_calls — already structured, no parsing needed.
    """
    return llm_response.text, llm_response.tool_calls


def extract_actions_from_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """
    Extract actions from raw tool_calls list (used by streaming handler).
    Tool calls are already structured dicts with 'type' key from generate_stream.
    """
    return [tc for tc in tool_calls if tc.get("type")]
