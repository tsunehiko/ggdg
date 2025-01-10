import argparse

FLAGS = argparse.Namespace()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_shot", type=int, default=3)
    parser.add_argument("--test_num", type=int, default=-1)
    parser.add_argument("--game_list_path", type=str, default=None)

    # llm
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--freq_penalty", type=float, default=0.0) # for gpt
    parser.add_argument("--repetition_penalty", type=float, default=1.0) # for huggingface
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_max_tokens", type=int, default=5000)
    parser.add_argument("--input_max_tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", type=str, default=None) # for sft model
    parser.add_argument("--load_in_4bit", action="store_true") # for hf model
    parser.add_argument("--load_in_8bit", action="store_true") # for hf model
    
    # grammar llm (optional)
    parser.add_argument("--grammar_engine", type=str, default=None)
    parser.add_argument("--grammar_temperature", type=float, default=0.0)
    parser.add_argument("--grammar_freq_penalty", type=float, default=0.0)
    parser.add_argument("--grammar_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--grammar_top_p", type=float, default=0.9)
    parser.add_argument("--grammar_output_max_tokens", type=int, default=5000)
    parser.add_argument("--grammar_input_max_tokens", type=int, default=8192)
    parser.add_argument("--grammar_tokenizer", type=str, default=None)
    parser.add_argument("--grammar_load_in_4bit", action="store_true")
    parser.add_argument("--grammar_load_in_8bit", action="store_true")

    # decoding
    parser.add_argument("--prompt_mode", type=str, required=True)
    parser.add_argument("--random_sampling", action="store_true")
    parser.add_argument("--use_oracle_rule_flag", action="store_true")
    parser.add_argument("--constrain_rule_gen_flag", action="store_true")
    parser.add_argument("--constrain_prog_gen_flag", action="store_true")
    parser.add_argument("--gd_max_num_correction", type=int, default=10)
    parser.add_argument("--r_max_num_correction", type=int, default=20)
    
    # evaluation
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_mode", type=str, default="full")

    args = parser.parse_args()
    FLAGS.__dict__.update(args.__dict__)
