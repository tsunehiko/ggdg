import abc
import os
import pickle
import hashlib
from pathlib import Path
from typing import List
from dataclasses import dataclass

import torch

from ggdg.train_utils import logger
from ggdg.structs import LLMResponse


@dataclass
class LLMConfig:
    engine: str
    tokenizer: str
    cache_dir: str
    temperature: float
    input_max_tokens: int
    output_max_tokens: int
    top_p: float
    repetition_penalty: float
    freq_penalty: float
    load_in_4bit: bool
    load_in_8bit: bool
    platform: str = ""
    model_name: str = ""


def str_to_identifier(x: str) -> str:
    """Convert a string to a small string with negligible collision probability
    and where the smaller string can be used to identifier the larger string in
    file names.

    Importantly, this function is deterministic between runs and between
    platforms, unlike python's built-in hash function.

    References:
        https://stackoverflow.com/questions/45015180
        https://stackoverflow.com/questions/5297448
    """
    return hashlib.md5(x.encode('utf-8')).hexdigest()


class LargeLanguageModel(abc.ABC):
    """A pretrained large language model.""" 
    def __init__(self, args: LLMConfig) -> None:
        self.max_memory_allocated = 0.
        self.max_memory_reserved = 0.
        self.cache_dir = args.cache_dir
        self.args = args

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this LLM.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(self,
                            prompt: str,
                            temperature: float,
                            stop_token: str,
                            num_completions: int = 1) -> List[LLMResponse]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")
    
    @abc.abstractmethod
    def choice(self, sys_prompt: str, user_prompt: str, options: list, temperature: float) -> str:
        raise NotImplementedError("Override me!")
    
    @abc.abstractmethod
    def get_input_ids(self, sys_prompt: str, user_prompt: str) -> torch.Tensor:
        raise NotImplementedError("Override me!")

    def sample_completions(self,
                           sys_prompt: str,
                           user_prompt: str,
                           temperature: float,
                           stop_token: str,
                           num_completions: int = 1,
                           disable_cache: bool = False) -> List[LLMResponse]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for LLMs where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """
        all_prompt = f"{sys_prompt}\n{user_prompt}"

        # Set up the cache file.
        os.makedirs(self.cache_dir, exist_ok=True)
        llm_id = self.get_id()
        prompt_id = str_to_identifier(all_prompt)
        # If the temperature is 0, the seed does not matter.
        config_id = "_".join([f"{k}_{v}" for k, v in self.args.__dict__.items() 
                              if k in ["engine", "temperature", "top_p"]]).replace("/", "_")
        cache_filename = f"{config_id}_{prompt_id}.pkl"
        cache_filepath = Path(self.cache_dir) / cache_filename
        if not os.path.exists(cache_filepath):
            os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        if disable_cache or not os.path.exists(cache_filepath):
            logger.debug(f"Querying LLM {llm_id} with new prompt.")
            completions = self._sample_completions(sys_prompt,
                                                   user_prompt,
                                                   temperature,
                                                   stop_token, num_completions)
            # Cache the completions.
            with open(cache_filepath, 'wb') as f:
                pickle.dump(completions, f)
            logger.debug(f"Saved LLM response to {cache_filepath}.")

        # Load the saved completion.
        with open(cache_filepath, 'rb') as f:
            completions = pickle.load(f)
        logger.debug(f"Loaded LLM response from {cache_filepath}.")
        return completions

    def greedy_completion(self,
                          sys_prompt: str,
                          user_prompt: str,
                          stop_token: str,
                          disable_cache: bool = False) -> LLMResponse:
        """Sample a greedy completion from a prompt."""
        responses = self.sample_completions(sys_prompt, user_prompt, 0.0, stop_token, disable_cache=disable_cache)
        assert len(responses) == 1
        return responses[0]

    def update_gpu_memory_usage(self) -> None:
        self.max_memory_allocated = max(self.max_memory_allocated, torch.cuda.max_memory_allocated())
        self.max_memory_reserved = max(self.max_memory_reserved, torch.cuda.max_memory_reserved())
    
    def get_max_memory_stats(self):
        return {
            "max_allocated": self.max_memory_allocated / (1024 ** 2), # MB
            "max_reserved": self.max_memory_reserved / (1024 ** 2)  # MB
        }

    #@abc.abstractmethod
    def _sample_next_token_with_logit_bias(self, prompt, logit_bias, temperature):
        """Sample the next token from the model with a logit bias."""
        raise NotImplementedError("Override me!")
