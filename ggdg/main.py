import json
import random
import re
import tqdm
import time
import collections
import gc
from pathlib import Path
from textwrap import dedent

import numpy as np
import torch

from minEarley.parser import EarleyParser

from ggdg.flags import FLAGS, parse_args
from ggdg.evaluate import evaluate
from ggdg.dataset import load_sempar_data, load_sem_parser
from llms.models.llm import LLMConfig
from llms.models.utils import setup_llm, load_prompts
from llms.inference import llm_iterative_completion
from ggdg.game_description_decoding import decode_game_description
from ggdg.random_sampling import randomly_decode_game_description
from ggdg.rule_decoding import decode_rule
from ggdg.train_utils import logger, setup_logger_file
from ggdg.utils import *


def save_results(log_dir, example_id, input_example, _prompt, llm_call_count_dict, ret_predictions, ret_grammars=None, demo_examples=None):
    prompt_dir = Path(log_dir) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prediction_program_dir = Path(log_dir) / "prediction" / "program"
    prediction_program_dir.mkdir(parents=True, exist_ok=True)
    prediction_grammar_dir = Path(log_dir) / "prediction" / "grammar"
    prediction_grammar_dir.mkdir(parents=True, exist_ok=True)
    prediction_info_dir = Path(log_dir) / "prediction" / "info"
    prediction_info_dir.mkdir(parents=True, exist_ok=True)
    gt_program_dir = Path(log_dir) / "gt" / "program"
    gt_program_dir.mkdir(parents=True, exist_ok=True)
    gt_grammar_dir = Path(log_dir) / "gt" / "grammar"
    gt_grammar_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = Path(log_dir) / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    error_dir = Path(log_dir) / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    
    with open(prompt_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
        f.write(_prompt)
    with open(prediction_program_dir / f"{example_id}_{input_example.gamename}.lud", "w") as f:
        if ret_predictions is not None:
            f.write(ret_predictions)
    with open(gt_program_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
        f.write(input_example.target)
    with open(gt_grammar_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
        f.write(input_example.grammar)
    if ret_grammars is not None:
        with open(prediction_grammar_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
            if ret_grammars is not None:
                f.write(ret_grammars)
    result_info = {}
    result_info["gamepath"] = input_example.gamepath
    result_info["llm_call_count"] = llm_call_count_dict
    with open(prediction_info_dir / f"{example_id}_{input_example.gamename}.json", "w") as f:
        json.dump(result_info, f)
    with open(demo_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
        f.write(demo_examples)


def standard_generate(
    llm,
    prompt_template,
    input_examples, 
    constrain_prog_gen_flag,
    num_shot,
    log_dir=None,
):
    """
    Args:
        overnight_flag: overnight lf needs special handling when using linearized tree
        fewshot_prompt: if None, construct the prompt from the template
    """
    error_dir = Path(log_dir) / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    
    all_generation_times = []
    for example_id, input_example in tqdm.tqdm(enumerate(input_examples), total=len(input_examples)):
        logger.info("-" * 80)
        logger.info(f"example {example_id} {input_example.gamename}")
        
        start_time = time.time()
        exemplars = input_example.train_examples
        examples_list = []
        for ex_id, exemplar in enumerate(exemplars):
            example_dict = {
                "query": exemplar.source,
                "program": dedent(exemplar.target),
            }
            examples_list.append(example_dict)

        pred_program = None
        task = {
            "query": input_example.source,
        }
        prog_user_prompt_dict = {
            "task": task,
        }
        if len(examples_list) > 0:
            prog_user_prompt_dict["examples"] = examples_list
        response_text = ""
        try:
            if constrain_prog_gen_flag:
                pred_program, llm_call_count_program = decode_game_description(
                    llm, prompt_template, prog_user_prompt_dict, 
                    local_parser=global_parser, global_parser=global_parser, 
                    max_num_correction=FLAGS.gd_max_num_correction, oracle_grammar=False)
            else:
                prog_user_prompt = dict_to_xml(prog_user_prompt_dict)
                if num_shot == 0:
                    sys_prompt_prog = prompt_template["system_prompt_template"]["zero-shot_program_generation"]
                else:
                    sys_prompt_prog = prompt_template["system_prompt_template"]["program_generation"]
                init_counter = collections.Counter()
                init_counter_key = sys_prompt_prog + prog_user_prompt
                program_xml_pattern = re.compile(r'<program>(.*?)</program>', re.DOTALL)
                pred_program, response_text, init_counter, init_llm_call_count = llm_iterative_completion(
                    llm, sys_prompt_prog, prog_user_prompt, init_counter, init_counter_key, 3, 3, program_xml_pattern)
                llm_call_count_program = init_llm_call_count
        except Exception as e:
            error_txt = f"Program Generation Error\n\nError: \n```\n{e}\n```\n\nPrediction:\n```\n{str(pred_program)}\n```"
            if response_text != "":
                error_txt += f"\n\nResponse:\n```\n{response_text}\n```"
            with open(error_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
                f.write(error_txt)
        all_generation_times.append(time.time() - start_time)
        
        llm_call_count_dict = {"rule": 0, "program": llm_call_count_program, "total": llm_call_count_program}
        save_results(log_dir, example_id, input_example, json.dumps(prog_user_prompt_dict), llm_call_count_dict, pred_program, None, 
                    dict_to_xml({"examples": examples_list}))
        logger.info("-" * 30 + " Summary " + "-" * 30)
        logger.info(f"    source:\n{input_example.source}")
        logger.info(f"prediction:\n{pred_program}")
        logger.info(f"    target:\n{input_example.target}")
        logger.info("-" * 70)
    
    return {
        "all_generation_times": all_generation_times,
    }

def grammar_based_generate(
        program_llm,
        grammar_llm,
        prompt_template,
        input_examples,
        random_sampling,
        use_oracle_rule_flag,
        constrain_rule_gen_flag,
        constrain_prog_gen_flag,
        num_shot=3,
        log_dir=None,
    ):
    """
    Args:
        use_oracle_rule_flag: if True, use oracle rule to generate the prompt
        constrain_rule_gen_flag: if True, constrain rule generation
        constrain_prog_gen_flag: if True, constrain program generation
    """
    error_dir = Path(log_dir) / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    
    all_generation_times, rule_generation_times, program_generation_times = [], [], []
    for example_id, input_example in tqdm.tqdm(enumerate(input_examples), total=len(input_examples)):
        logger.info("=" * 80)
        logger.info(f"example {example_id} {input_example.gamename}")
        llm_call_count = 0
        
        start_time = time.time()
        exemplars = input_example.train_examples
        examples_list = []
        for ex_id, exemplar in enumerate(exemplars):
            example_dict = {
                "bnf_grammar_rules": dedent(exemplar.grammar),
                "query": exemplar.source,
                "program": dedent(exemplar.target),
            }
            examples_list.append(example_dict)
        
        llm_call_count_rule = 0
        rule_start_time = time.time()
        if use_oracle_rule_flag:
            logger.info("Use oracle rule")
            input_example.pred_grammar = input_example.grammar
            pred_bnf_grammar = input_example.grammar
            complement_suffix = ""
        else:
            pred_bnf_grammar, complement_suffix = None, ""
            task = {
                "query": input_example.source,
            }
            rule_user_prompt_dict = {
                "task": task,
            }
            if len(examples_list) > 0:
                rule_user_prompt_dict["examples"] = examples_list
            try:
                if constrain_rule_gen_flag:
                    pred_bnf_grammar, llm_call_count_rule, complement_suffix = decode_rule(
                        grammar_llm, prompt_template, rule_user_prompt_dict, global_rules, raw_global_rules, 
                        constraint_mode=grammar_llm.args.platform in ["hf", "local"], 
                        max_num_correction=FLAGS.r_max_num_correction, num_shot=num_shot)
                else:
                    rule_user_prompt = dict_to_xml(rule_user_prompt_dict)
                    if num_shot == 0:
                        sys_prompt_rule = prompt_template["system_prompt_template"]["zero-shot_grammar_generation"]
                    else:
                        sys_prompt_rule = prompt_template["system_prompt_template"]["grammar_generation"]
                    response = grammar_llm.sample_completions(sys_prompt_rule, rule_user_prompt, grammar_llm.args.temperature, stop_token="\n\n")[0]
                    bnf_xml_pattern = re.compile(r'<bnf_grammar_rules>(.*?)</bnf_grammar_rules>', re.DOTALL)
                    pred_bnf_grammar = [match for match in bnf_xml_pattern.findall(response.response_text)][0].strip()
                    llm_call_count_rule = 1
                llm_call_count += llm_call_count_rule
            except Exception as e:
                with open(error_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
                    f.write(f"Rule Generation Error\n\nError: \n```\n{e}\n```\n\nPrediction:\n```\n{str(pred_bnf_grammar)}\n```")
            input_example.pred_grammar = pred_bnf_grammar
        rule_generation_times.append(time.time() - rule_start_time)

        # post-process grammar
        if pred_bnf_grammar is not None and len(pred_bnf_grammar) > 0:
            pred_lark_grammar = bnf2lark(pred_bnf_grammar)
            if complement_suffix:
                lark_grammar_for_local = pred_lark_grammar + "\n" + bnf2lark(complement_suffix)
            else:
                lark_grammar_for_local = pred_lark_grammar
            logger.info(f"earley correction with grammar\n{lark_grammar_for_local}")
        else:
            lark_grammar_for_local = ""
        
        # program generation
        pred_program = None
        llm_call_count_program = 0
        program_start_time = time.time()
        if random_sampling:
            pred_program = randomly_decode_game_description(lark_grammar_for_local, use_oracle_rule_flag, global_parser)
            llm_call_count_program = 0
            prog_user_prompt_dict = {}
        else:
            task = {}
            if pred_bnf_grammar is not None and len(pred_bnf_grammar) > 0:
                task["bnf_grammar_rules"] = dedent(pred_bnf_grammar)
            task = {"query": input_example.source}
            prog_user_prompt_dict = {
                "examples": examples_list,
                "task": task,
            }
            try:
                if constrain_prog_gen_flag:
                    # create local parser
                    if use_oracle_rule_flag:
                        lark_grammar_for_local = decorate_oracle_grammar_ludii(lark_grammar_for_local)
                    else:
                        lark_grammar_for_local = decorate_grammar_ludii(lark_grammar_for_local, replace_prefix=grammar_llm.args.platform != "local")
                    try:
                        local_parser = EarleyParser(lark_grammar_for_local, start=global_parser.option.start, 
                                                    oracle_rule=use_oracle_rule_flag or grammar_llm.args.platform == "local")
                        logger.info(f"[Rule][Success] Local parser has been created with the predicted grammar!")
                    except Exception as e:
                        logger.warning(f"failed to create parser due to {e}, reverting to global parser")
                        local_parser = global_parser
                    
                    # generate program
                    prediction_program_dir = Path(log_dir) / "prediction" / "program"
                    prediction_program_dir.mkdir(parents=True, exist_ok=True)
                    prediction_program_path = prediction_program_dir / f"{example_id}_{input_example.gamename}.lud"
                    pred_program, llm_call_count_program = decode_game_description(program_llm, prompt_template, prog_user_prompt_dict, 
                                                                                    local_parser, FLAGS.gd_max_num_correction, 
                                                                                    use_oracle_rule_flag,
                                                                                    prediction_program_path, num_shot)
                else:
                    prog_user_prompt = dict_to_xml(prog_user_prompt_dict)
                    if num_shot == 0:
                        sys_prompt_prog = prompt_template["system_prompt_template"]["zero-shot_grammar-based_program_generation"]
                    else:
                        sys_prompt_prog = prompt_template["system_prompt_template"]["grammar-based_program_generation"]
                    response = program_llm.sample_completions(sys_prompt_prog, prog_user_prompt, program_llm.args.temperature, stop_token="\n\n")[0]
                    program_xml_pattern = re.compile(r'<program>(.*?)</program>', re.DOTALL)
                    pred_program = [match for match in program_xml_pattern.findall(response.response_text)][0].strip()
                    llm_call_count_program = 1
            except Exception as e:
                with open(error_dir / f"{example_id}_{input_example.gamename}.txt", "w") as f:
                    f.write(f"Program Generation Error\n\nError: \n```\n{e}\n```\n\nPrediction:\n```\n{str(pred_program)}\n```")
        llm_call_count += llm_call_count_program
        all_generation_times.append(time.time() - start_time)
        program_generation_times.append(time.time() - program_start_time)
        
        llm_call_count_dict = {"rule": llm_call_count_rule, "program": llm_call_count_program, "total": llm_call_count}
        save_results(log_dir, example_id, input_example, json.dumps(prog_user_prompt_dict), llm_call_count_dict, pred_program, pred_bnf_grammar, 
                     dict_to_xml({"examples": examples_list}))

        logger.info("-" * 30 + " Summary " + "-" * 30)
        logger.info(f"source:\n{input_example.source}")
        logger.info(f"prediction:\n{pred_program}")
        logger.info(f"target:\n{input_example.target}")
        logger.info(f"predicted grammar:\n{pred_bnf_grammar}")
        logger.info(f"parser grammar:\n{lark_grammar_for_local}")
        logger.info(f"oracle grammar:\n{input_example.grammar}")
        logger.info("-" * 70)
    
    return {
        "all_generation_times": all_generation_times,
        "rule_generation_times": rule_generation_times,
        "program_generation_times": program_generation_times,
    }


if __name__ == "__main__":
    # setup
    parse_args()
    random.seed(FLAGS.seed)
    config = vars(FLAGS)
    exp_name = config["exp_name"]
    log_dir = f"log/{exp_name}"
    setup_logger_file(logger, log_dir)
    logger.info(config)

    ## setup grammar and parser
    global_parser, global_rules, raw_global_rules = load_sem_parser(config)

    ## setup program llm
    llm_cache_dir = Path(log_dir) / "llm_cache"
    program_llm_config = LLMConfig(
        engine=FLAGS.engine,
        tokenizer=FLAGS.tokenizer,
        cache_dir=llm_cache_dir,
        temperature=FLAGS.temperature,
        freq_penalty=FLAGS.freq_penalty,
        repetition_penalty=FLAGS.repetition_penalty,
        top_p=FLAGS.top_p,
        output_max_tokens=FLAGS.output_max_tokens,
        input_max_tokens=FLAGS.input_max_tokens,
        load_in_4bit=FLAGS.load_in_4bit,
        load_in_8bit=FLAGS.load_in_8bit,
    )
    program_llm = setup_llm(program_llm_config)
    logger.info("=" * 40 + " Program LLM info " + "=" * 40)
    logger.info(f"LLM: {program_llm_config.engine}")
    logger.info(f"Tokenizer: {program_llm_config.tokenizer}")
    logger.info(f"Load in 4bit: {program_llm_config.load_in_4bit} Load in 8bit: {program_llm_config.load_in_8bit}")
    
    # load grammar llm
    if FLAGS.grammar_engine is None:
        grammar_llm = program_llm
    else:
        grammar_llm_cache_dir = Path(log_dir) / "grammar_llm_cache"
        grammar_llm_config = LLMConfig(
            engine=FLAGS.grammar_engine,
            tokenizer=FLAGS.grammar_tokenizer,
            cache_dir=grammar_llm_cache_dir,
            temperature=FLAGS.grammar_temperature,
            freq_penalty=FLAGS.grammar_freq_penalty,
            repetition_penalty=FLAGS.grammar_repetition_penalty,
            top_p=FLAGS.grammar_top_p,
            output_max_tokens=FLAGS.grammar_output_max_tokens,
            input_max_tokens=FLAGS.grammar_input_max_tokens,
            load_in_4bit=FLAGS.grammar_load_in_4bit,
            load_in_8bit=FLAGS.grammar_load_in_8bit,
        )
        grammar_llm = setup_llm(grammar_llm_config)
        logger.info("=" * 40 + " Grammar LLM info " + "=" * 40)
        logger.info(f"LLM: {grammar_llm_config.engine}")
        logger.info(f"Tokenizer: {grammar_llm_config.tokenizer}")
        logger.info(f"Load in 4bit: {grammar_llm_config.load_in_4bit} Load in 8bit: {grammar_llm_config.load_in_8bit}")
    
    ## load prompts
    prompt_template = load_prompts(FLAGS.engine)
    prompt_dir = Path(log_dir) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    with open(prompt_dir / "prompt_template.json", "w") as f:
        json.dump(prompt_template, f, indent=4)

    # load data
    examples = load_sempar_data(config)
    logger.info(f"{len(examples)} test examples")

    # generate
    if  config["prompt_mode"] == "std":
        results = standard_generate(
            program_llm, prompt_template, examples, 
            constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, num_shot=FLAGS.num_shot, log_dir=log_dir)
        test_grammar_counters = None
    else:
        results = grammar_based_generate(
            program_llm, grammar_llm, prompt_template, examples, random_sampling=FLAGS.random_sampling, 
            use_oracle_rule_flag=FLAGS.use_oracle_rule_flag, 
            constrain_rule_gen_flag=FLAGS.constrain_rule_gen_flag, 
            constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, 
            num_shot=FLAGS.num_shot, log_dir=log_dir)
    
    # gpu stats (program llm)
    gpu_stats = program_llm.get_max_memory_stats()
    results["gpu_max_memory_allocated"] = gpu_stats["max_allocated"]
    results["gpu_max_memory_reserved"] = gpu_stats["max_reserved"]
    
    # log results
    logger.info("=" * 40 + " Generation info " + "=" * 40)
    for key, value in results.items():
        if isinstance(value, list):
            logger.info(f"{key}: {np.mean(value)} +/- {np.std(value)}")
        else:
            logger.info(f"{key}: {value}")
    
    # clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    # evaluate
    if FLAGS.eval:
        evaluate(Path(log_dir), FLAGS.eval_mode)
