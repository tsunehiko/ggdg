
import time
import os
import enum
from typing import Any, Dict, List

import torch
import instructor
from pydantic import BaseModel, create_model
import openai
import tiktoken

from ggdg.structs import LLMResponse
from llms.models.llm import LargeLanguageModel, LLMConfig


class ChatGPT(LargeLanguageModel):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    def __init__(self, args: LLMConfig) -> None:
        super().__init__(args)
        self._model_name = args.model_name
        self.client = openai.OpenAI()
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.instructor_client = instructor.from_openai(self.client)

    def get_id(self) -> str:
        return f"chatgpt/{self._model_name}"
    
    def get_input_ids(self, sys_prompt: str, user_prompt: str) -> torch.Tensor:
        tokens = self.token_encoder.encode(sys_prompt + "\n" + user_prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        return tokens

    def _sample_completions(
            self,
            sys_prompt: str,
            user_prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:  
        """
        Note that sys and user prompt are assumed to be separated by a newline.
        """
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
            ]

        response = None
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    stop=stop_token,
                    max_tokens=self.args.output_max_tokens,
                    frequency_penalty=self.args.freq_penalty,
                    top_p=self.args.top_p,
                    n=num_completions)
                # Successfully queried, so break.
                break
            except (openai.RateLimitError,
                    openai.APIConnectionError, openai.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query OpenAI API.")

        assert len(response.choices) == num_completions
        return [
            self._raw_to_llm_response(r, messages, temperature,stop_token, num_completions)
            for r in response.choices
        ]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             messages: list,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        # text = raw_response["message"]["content"]
        text = raw_response.message.content
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(messages,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())
    
    def choice(self, sys_prompt: str, user_prompt: str, options: list, temperature: float) -> str:
        labels = enum.Enum("Label", {option: option for option in options})
        Options = create_model(
            "Options",
            class_label=(labels, ...),
            __base__=BaseModel,
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
            ]
        response = None
        
        for _ in range(5):
            try:
                response = self.instructor_client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    response_model=Options,
                    temperature=temperature,
                )
                break
            except (openai.RateLimitError,
                    openai.APIConnectionError, openai.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)
        
        choice = response.class_label.value
        return choice
