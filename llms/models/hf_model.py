
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import outlines
from outlines.samplers import multinomial

from ggdg.train_utils import logger
from ggdg.structs import LLMResponse
from llms.models.llm import LargeLanguageModel, LLMConfig


class HFModel(LargeLanguageModel):
    def __init__(self, args: LLMConfig) -> None:
        super().__init__(args)
        self._model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model_kwargs = {
            "pretrained_model_name_or_path": self._model_name,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,)
        elif args.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.args = args

    def get_id(self) -> str:
        return f"hf_{self._model_name}"
    
    def get_input_ids(self, sys_prompt: str, user_prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        return input_ids

    def _sample_completions(
            self,
            sys_prompt: str,
            user_prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        with torch.no_grad():
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            logger.debug(f"input length: {input_ids.shape[-1]}")
            if input_ids.shape[-1] > self.args.input_max_tokens:
                raise ValueError("Input length exceeds the maximum token length.")
            
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            input_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": self.args.output_max_tokens,
                "eos_token_id": terminators,
                "do_sample": True,
                "top_p": self.args.top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                input_kwargs["temperature"] = temperature
            if self.args.repetition_penalty > 1.0:
                input_kwargs["repetition_penalty"] = self.args.repetition_penalty
            
            response = self.model.generate(**input_kwargs)
            
            self.update_gpu_memory_usage()
            torch.cuda.empty_cache()

        if response is None:
            raise RuntimeError("Failed to query.")

        assert len(response) == num_completions
        return [
            self._raw_to_llm_response(r[input_ids.shape[-1]:], messages, temperature, stop_token, num_completions)
            for r in response
        ]

    def _raw_to_llm_response(self,
                             raw_response: torch.Tensor,
                             messages: list, 
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        
        raw_text = self.tokenizer.decode(raw_response, skip_special_tokens=True)
        text = raw_text.strip()
        
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(messages,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response)
    
    def choice(self, sys_prompt: str, user_prompt: str, options: list, temperature: float) -> str:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        with torch.no_grad():
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True, 
                tokenize=False
            )
            
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            terminators = self.tokenizer.convert_ids_to_tokens(terminators)
            
            outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
            sampler_input_kwargs = {"top_p": self.args.top_p}
            if temperature > 0:
                sampler_input_kwargs["temperature"] = temperature
            generator = outlines.generate.choice(outlines_model, options, sampler=multinomial(**sampler_input_kwargs))
            response = generator(input_ids, max_tokens=self.args.output_max_tokens, stop_at=terminators)
            
            torch.cuda.empty_cache()

        if response is None:
            raise RuntimeError("Failed to query")

        return response
