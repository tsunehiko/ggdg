from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class LLMResponse:
    """A single response from a LargeLanguageModel."""
    prompt_text: list
    response_text: str
    prompt_info: Dict
    other_info: Dict

@dataclass(frozen=True)
class DecodingRecord:
    system_prompt: str
    user_prompt: str
    response: str
    prediction: str
