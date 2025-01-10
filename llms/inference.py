import collections
from typing import Union

from ggdg.train_utils import logger
from ggdg.utils import *
from llms.models.llm import LargeLanguageModel


def llm_iterative_completion(llm: LargeLanguageModel, system_prompt: str, user_prompt: str, 
                            counter: collections.Counter, counter_key: str, call_limit: int, repetition_limit: int,
                            parse_pattern = None, disable_cache: bool = False) -> Union[str, str, collections.Counter, int]:
    llm_call_count = 0
    prediction = ""
    response_text = ""
    while llm_call_count < call_limit:
        if counter_key in counter:
            if counter[counter_key] > repetition_limit:
                logger.error("[Interruption] Repetition limit exceeded.")
                break
            counter[counter_key] += 1
            _system_prompt = system_prompt + f"\n(Retry: {counter[counter_key]})\n"
        else:
            counter[counter_key] = 1
            _system_prompt = system_prompt
        _user_prompt = user_prompt
        
        try:
            response = llm.greedy_completion(_system_prompt, _user_prompt, stop_token="\n\n", disable_cache=disable_cache)
            response_text = response.response_text
            logger.debug(f"Response:\n {response_text}")
            if parse_pattern is None:
                prediction = response_text
            else:
                prediction = [match for match in parse_pattern.findall(response_text)][0].strip()
        except Exception as e:
            logger.error(e)
            counter[counter_key] -= 1
        
        llm_call_count += 1
        if prediction != "":
            break
    return prediction, response_text, counter, llm_call_count


def llm_iterative_choice(llm: LargeLanguageModel, system_prompt: str, user_prompt: str, options: list[str],
                            counter: collections.Counter, counter_key: str, call_limit: int, repetition_limit: int,
                            ) -> Union[str, collections.Counter, int]:
    llm_call_count = 0
    prediction = ""
    while llm_call_count < call_limit:
        if counter_key in counter:
            if counter[counter_key] > repetition_limit:
                logger.error("[Interruption] Repetition limit exceeded.")
                break
            counter[counter_key] += 1
            _system_prompt = system_prompt + f"\n(Retry: {counter[counter_key]})\n"
        else:
            counter[counter_key] = 1
            _system_prompt = system_prompt
        _user_prompt = user_prompt
        
        try:
            prediction = llm.choice(_system_prompt, _user_prompt, options, temperature=0.0)
        except Exception as e:
            logger.error(e)
        
        prediction = prediction.replace("\\", "")
        llm_call_count += 1
        if prediction != "":
            break
    return prediction, counter, llm_call_count
