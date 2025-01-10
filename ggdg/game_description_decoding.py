import collections
import copy
import subprocess
import re
import json
from typing import Union
from dataclasses import dataclass

from ggdg.structs import DecodingRecord
from ggdg.train_utils import logger
from ggdg.utils import *
from llms.models.llm import LargeLanguageModel
from llms.inference import llm_iterative_completion, llm_iterative_choice


MAX_REPETITION = 3
MAX_LOCAL_NUM_CORRECTION = 3


@dataclass
class ProgramDecodingRecord:
    terminal_record: DecodingRecord
    program_record: DecodingRecord
    prefix: str
    suffix: list


def obtain_program_correction_pairs(prediction, parser):
    """
    Returns a list of candidates in the form of (prefix, suffix).
    """
    try:
        parser.parse(prediction)
        return []
    except Exception as runtime_e:
        return parser.handle_error(runtime_e)


def escape_special_characters(input_string: str) -> str:
    special_characters = r'[\.\^\$\*\+\?\{\}\[\]\(\)\\\|]'
    escaped_string = re.sub(special_characters, r'\\\g<0>', input_string)
    return escaped_string


def get_constant_regex(option):
    if option == "FLOAT_CONSTANT":
        return '[-]?\d+ | [-]?\d*\.\d+([eE][-]?\d+)?'
    elif option == "INT_CONSTANT":
        return '[-]?\d+'
    elif option == "STRING":
        return '"[^"]*"'
    else:
        return option


def get_terminal_input_suffix(suffix: list[str], llm: LargeLanguageModel, oracle_grammar: bool = False) -> Union[list[str], dict[str, list[str]]]:
    processed_suffix = [escape_special_characters(s) for s in suffix]
    if llm.args.platform in ["hf", "local"] and not oracle_grammar:
        processed_suffix = [get_constant_regex(s) for s in processed_suffix]
    return processed_suffix


def get_program_system_prompt(terminal_prediction: str, prompt_template: dict, num_shot: int = 3) -> str:
    if num_shot == 0:
        program_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_program_correction"]
    else:
        program_sys_prompt = prompt_template["system_prompt_template"]["program_correction"]
    # 予測した終端記号がSTRING, INT_CONSTANT, FLOAT_CONSTANTの場合、それに対応するプロンプトを追加
    if "STRING" in terminal_prediction:
        program_sys_prompt += prompt_template["string_prompt"]
    if "INT_CONSTANT" in terminal_prediction:
        program_sys_prompt += prompt_template["int_prompt"]
    if "FLOAT_CONSTANT" in terminal_prediction:
        program_sys_prompt += prompt_template["float_prompt"]
    return program_sys_prompt


def get_min_prefix(prefix) -> bool:
    return re.sub(r'\s{2,}', ' ', prefix.replace("\n", "")).strip()


def get_back_record(records: list[ProgramDecodingRecord]):
    if len(records) == 0:
        return "", []
    terminal_prediction = records[-1].terminal_record.prediction
    if len(records[-1].suffix) == 0:
        if len(records) == 1:
            return "", []
        else:
            _ = records.pop()
            prefix, suffix = get_back_record(records)
    else:
        if terminal_prediction not in ["STRING", "INT_CONSTANT", "FLOAT_CONSTANT"] and \
            terminal_prediction in records[-1].suffix:
            records[-1].suffix.remove(terminal_prediction)
        else:
            match = re.compile(r'([-]?\d+|[-]?\d*\.\d+([eE][-]?\d+)?|"[^"]*")').match(terminal_prediction)
            if not match:
                logger.error(f"Invalid terminal prediction: {terminal_prediction}")
        if len(records[-1].suffix) == 0:
            _ = records.pop()
            prefix, suffix = get_back_record(records)
        else:
            prefix, suffix = records[-1].prefix, records[-1].suffix
    return prefix, suffix


def repeat_obtain_program_correction_pairs(program, parser, max_repetition=20):
    pairs = ["", []]
    skip_count = collections.Counter()
    repetition_count = 0
    while repetition_count < 1000:
        pairs = obtain_program_correction_pairs(program, parser)
        if len(pairs) == 0:
            return program, []
        prefix, suffix = pairs[0].strip(), list(set(pairs[1]))
        if len(suffix) == 1 and suffix[0] not in ["STRING", "INT_CONSTANT", "FLOAT_CONSTANT"]:
            logger.debug("[Program][Processing] Skip terminal prediction")
            logger.debug(f"[Program] prefix: \n{prefix}")
            logger.debug(f"[Program] suffix: {suffix}")
            program = prefix + " " + suffix[0]
            skip_count[suffix[0]] += 1
            if skip_count[suffix[0]] > max_repetition:
                logger.info("[Program] Loop detected")
                return "", []
            repetition_count += 1
        else:
            return prefix, suffix


def decode_game_description(llm, prompt_template, program_user_prompt_dict, local_parser, 
                            max_num_correction=30, oracle_grammar=False, prediction_path=None, num_shot=3):
    llm_call_count = 0
    records: list[ProgramDecodingRecord] = []
    
    # Initial prediction
    if num_shot == 0:
        init_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_grammar-based_program_generation"]
    else:
        init_sys_prompt = prompt_template["system_prompt_template"]["grammar-based_program_generation"]
    init_user_prompt = dict_to_xml(program_user_prompt_dict)
    init_counter = collections.Counter()
    init_counter_key = init_sys_prompt + init_user_prompt
    program_xml_pattern = re.compile(r'<program>(.*?)</program>', re.DOTALL)
    program_prediction, response_text, init_counter, init_llm_call_count = llm_iterative_completion(
        llm, init_sys_prompt, init_user_prompt, init_counter, init_counter_key, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION, program_xml_pattern)
    llm_call_count += init_llm_call_count
    logger.info("-" * 20)
    logger.info("[Program] Initial prediction")
    logger.info(f"program_prediction: \n{program_prediction}")
    initial_prediction = ""
    is_compilable_init_prediction = False
    if program_prediction != "":
        with open(prediction_path, "w") as f:
            f.write(program_prediction)
        try:
            result = subprocess.run(['java', '-jar', 'ludii_java/jars/EvalLudiiGame.jar', '--game-path', str(prediction_path),
                                     ], capture_output=True, text=True, timeout=300)
            result_str = re.findall(r"\{'isCompilable'.*?\}", result.stdout)[0]
        except:
            result_str = "{'isCompilable': 'false'}"
        result_dict = json.loads(result_str.strip().replace("'", '"'))
        logger.info(f"[Program] Initial prediction result: {result_dict}")
        is_compilable_init_prediction = result_dict["isCompilable"] == "true"
        if is_compilable_init_prediction:
            logger.info("*" * 20)
            logger.info("[Program][Success][Global] Complete program found")
            return program_prediction, llm_call_count
        else:
            initial_prediction = program_prediction
    
    correction_count = 0
    terminal_counter, program_counter = collections.Counter(), collections.Counter()
    is_completed, is_interrupted, loop_detected = False, False, False
    while correction_count < max_num_correction:
        logger.debug("-" * 40)
        logger.debug(f'[Program] Correction Count: {correction_count}')
        correction_count += 1
        
        prefix, suffix = repeat_obtain_program_correction_pairs(program_prediction, local_parser)
        if len(prefix) == 0:
            loop_detected = True
        elif len(suffix) == 0:
            program_prediction = prefix
            logger.info("*" * 20)
            logger.info("[Program][Success][Local] Complete program found")
            is_completed = True
            break

        if len(records) == 0:
            records.append(ProgramDecodingRecord(None, None, prefix, suffix))
        else:
            min_prefix, min_prev_prefix = get_min_prefix(prefix), get_min_prefix(records[-1].prefix)
            if min_prefix == min_prev_prefix or len(min_prefix) < len(min_prev_prefix) or loop_detected:
                logger.debug("[Program][Get Back] Get back to the previous record")
                prefix, suffix = get_back_record(records)
                terminal_counter[prefix] = 1
                loop_detected = False
            else:
                records.append(ProgramDecodingRecord(None, None, prefix, suffix))
        if len(suffix) == 0:
            logger.debug("[Program] No candidates found")
            is_interrupted = True
            break
        input_suffix = get_terminal_input_suffix(suffix, llm, oracle_grammar=oracle_grammar)
        logger.debug(f"[Program] prefix: \n{prefix}")
        logger.debug(f"[Program] suffix: \n{input_suffix}")
        
        # terminal prediction
        terminal_program_user_prompt_dict = copy.deepcopy(program_user_prompt_dict)
        terminal_program_user_prompt_dict["task"]["terminal_candidates"] = suffix
        terminal_program_user_prompt_dict["task"]["program"] = prefix
        if num_shot == 0:
            terminal_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_terminal_selection"]
        else:
            terminal_sys_prompt = prompt_template["system_prompt_template"]["terminal_selection"]
        terminal_user_prompt = dict_to_xml(terminal_program_user_prompt_dict)
        terminal_user_prompt = terminal_user_prompt.replace("\n</program>\n</task>", "")
        terminal_prediction, terminal_counter, local_llm_call_count_for_terminal = llm_iterative_choice(
            llm, terminal_sys_prompt, terminal_user_prompt, input_suffix, terminal_counter, prefix, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION)
        llm_call_count += local_llm_call_count_for_terminal
        logger.debug(f"[Program] terminal_prediction: {terminal_prediction}")
        records[-1].terminal_record = DecodingRecord(terminal_sys_prompt, terminal_user_prompt, "", terminal_prediction)
        if terminal_counter[prefix] > MAX_REPETITION:
            is_interrupted = True
            break
        
        partial_program_prediction = prefix + " " + terminal_prediction
        prefix, suffix = repeat_obtain_program_correction_pairs(partial_program_prediction, local_parser)
        if len(prefix) == 0:
            loop_detected = True
            continue
        elif len(suffix) == 0:
            program_prediction = prefix
            logger.info("*" * 20)
            logger.info("[Program][Success][Local] Complete program found")
            is_completed = True
            break
        partial_program_prediction = prefix
        
        # program prediction
        program_sys_prompt = get_program_system_prompt(terminal_prediction, prompt_template, num_shot)
        correction_program_user_prompt_dict = copy.deepcopy(program_user_prompt_dict)
        correction_program_user_prompt_dict["task"]["partial_program"] = partial_program_prediction
        program_user_prompt = dict_to_xml(correction_program_user_prompt_dict)
        prompt_key = program_sys_prompt + "\n\n" + program_user_prompt
        program_prediction, response_text, program_counter, local_llm_call_count_for_program = llm_iterative_completion(
            llm, program_sys_prompt, program_user_prompt, program_counter, prompt_key, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION, program_xml_pattern)
        llm_call_count += local_llm_call_count_for_program
        logger.debug(f"[Program] program_prediction: \n{program_prediction}")
        records[-1].program_record = DecodingRecord(program_sys_prompt, program_user_prompt, response_text, program_prediction)
    
    if is_interrupted:
        logger.info("[Program][Failure] Interrupted due to redundant predictions")
        program_prediction = initial_prediction
        logger.info("[Program] Reverting to initial prediction")
    elif correction_count == max_num_correction and not is_completed:
        pairs = obtain_program_correction_pairs(program_prediction, local_parser)
        if len(pairs) == 0:
            logger.info("*" * 20)
            logger.info("[Program][Success][Local] Complete program found")
        else:
            logger.info("[Program][Failure] Correction limit exceeded")
            if is_compilable_init_prediction or program_prediction == "":
                program_prediction = initial_prediction
                logger.info("[Program] Reverting to initial prediction")
    
    logger.debug("-" * 40)
    logger.debug(f"[Program] correction_count: {correction_count}")
    logger.debug(f"[Program] llm_call_count: {llm_call_count}")
    
    return program_prediction, llm_call_count
