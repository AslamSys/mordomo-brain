"""
Action extraction from LLM response.
The LLM is instructed to append [ACTION: {...}] for detectable intents.
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"\[ACTION:\s*(\{.*?\})\]", re.DOTALL)


def extract_actions(text: str) -> tuple[str, list[dict]]:
    """
    Parse [ACTION: {...}] tags from LLM response.
    Returns (clean_text_without_action_tags, list_of_action_dicts).
    """
    actions = []
    for match in _ACTION_RE.finditer(text):
        try:
            action = json.loads(match.group(1))
            actions.append(action)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse action JSON: %s — %s", match.group(1), e)

    clean_text = _ACTION_RE.sub("", text).strip()
    return clean_text, actions
