import json
from argparse import ArgumentParser
from textwrap import dedent

from transformers import AutoTokenizer

from ggdg.dataset import load_ludii_example
from ggdg.utils import dict_to_xml


def create_dataset(args):
    dataset = []
    
    # Load data
    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)
    system_program_prompt = prompts["system_prompt_template"]["zero-shot_program_generation"]
    system_grammar_prompt = prompts["system_prompt_template"]["zero-shot_grammar_generation"]
    system_program_grammar_prompt = prompts["system_prompt_template"]["zero-shot_grammar-based_program_generation"]
    with open(args.test_game_path, "r") as f:
        test_game_dict = json.load(f)
    with open(args.gamelist_path, "r") as f:
        gamelist = [line.strip() for line in f.readlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    for gamepath in gamelist:
        if gamepath in test_game_dict:
            continue
        example = load_ludii_example(gamepath)
        grammar_token_length = tokenizer(example.grammar, return_tensors="pt")["input_ids"].shape[1]
        program_token_length = tokenizer(example.target, return_tensors="pt")["input_ids"].shape[1]
        if grammar_token_length > args.max_token_len or program_token_length > args.max_token_len:
            continue
        if args.mode == "grammar":
            grammar_data = {"messages": [{"role": "system", "content": system_grammar_prompt},
                                        {"role": "user", "content": dict_to_xml({"task": {"query": example.source}})},
                                        {"role": "assistant", "content": dict_to_xml({"bnf_grammar_rules": dedent(example.grammar)})}]}
            dataset.append(grammar_data)
        elif args.mode == "program":
            program_data = {"messages": [{"role": "system", "content": system_program_prompt},
                                        {"role": "user", "content": dict_to_xml({"task": {"query": example.source}})},
                                        {"role": "assistant", "content": dict_to_xml({"program": dedent(example.target)})}]}
            dataset.append(program_data)
        elif args.mode == "grammar-based-program":
            program_grammar_data = {
                "messages": 
                    [
                        {"role": "system", "content": system_program_grammar_prompt},
                        {"role": "user", "content": dict_to_xml({"task": {"bnf_grammar_rules": dedent(example.grammar), "query": example.source}})},
                        {"role": "assistant", "content": dict_to_xml({"program": dedent(example.target)})}
                    ]
                }
            dataset.append(program_grammar_data)
        elif args.mode == "grammar-based-program-wo-query":
            program_grammar_data = {
                "messages": 
                    [
                        {"role": "system", "content": system_program_grammar_prompt},
                        {"role": "user", "content": dict_to_xml({"bnf_grammar_rules": dedent(example.grammar)})},
                        {"role": "assistant", "content": dict_to_xml({"program": dedent(example.target)})}
                    ]
                }
            dataset.append(program_grammar_data)
        elif args.mode == "both":
            grammar_data = {"messages": [{"role": "system", "content": system_grammar_prompt},
                                        {"role": "user", "content": dict_to_xml({"task": {"query": example.source}})},
                                        {"role": "assistant", "content": dict_to_xml({"bnf_grammar_rules": dedent(example.grammar)})}]}
            dataset.append(grammar_data)
            program_data = {"messages": [{"role": "system", "content": system_program_prompt},
                                        {"role": "user", "content": dict_to_xml({"task": {"query": example.source, "bnf_grammar_rules": dedent(example.grammar)}})},
                                        {"role": "assistant", "content": dict_to_xml({"program": dedent(example.target)})}]}
            dataset.append(program_data)
        else:
            raise NotImplementedError(f"mode {args.mode} not supported")
    
    print(f"Created dataset with {len(dataset)} examples")
    with open(args.output_dataset_path, "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--prompt_path', type=str, default="./prompts/default/prompts.json")
    args.add_argument('--output_dataset_path', type=str, default="./sft/data/program.json")
    args.add_argument('--gamelist_path', type=str, default="./data/ludii/gamelist_grammar.txt")
    args.add_argument('--test_game_path', type=str, default="./data/ludii/game_dicts/main.json")
    args.add_argument('--mode', type=str, default="program")
    args.add_argument('--tokenizer', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args.add_argument('--max_token_len', type=int, default=2000)
    args = args.parse_args()

    create_dataset(args)
