import collections
import copy
from textwrap import dedent
from dataclasses import dataclass

from ggdg.structs import DecodingRecord
from ggdg.train_utils import logger
from ggdg.utils import *
from llms.inference import llm_iterative_completion, llm_iterative_choice


MAX_NUM_CORRECTION = 20
MAX_LOCAL_NUM_CORRECTION = 3
MAX_TOKEN_LEN = 8192
MAX_REPETITION = 3


@dataclass(frozen=True)
class RuleDecodingRecord:
    record: DecodingRecord
    prefix: str
    suffix: list


def detect_loop(lhs, rhs):
    if len(rhs) == 0:
        return True
    elif len(rhs) > 1:
        target_rhs = []
        for r in rhs:
            if "(" not in r and ")" not in r and "{" not in r and "}" not in r:
                target_rhs.append(r)
        target_rhs = list(set(target_rhs))
    else:
        target_rhs = rhs
    r = target_rhs[0]
    if lhs == r or f'"({lhs})"'== r:
        return True
    else:
        return False


def find_child_nonterminals(nonterminal, ruleset, existing_nonterminals=[]):
    for rule in ruleset:
        if rule.origin == nonterminal:
            lhs, rhs = rule.origin, rule.expansion
            for r in rhs:
                if not re.match(r'^".*"$', r):
                    if r[-1] in ["+", "?"]:
                        origin = r[:-1]
                    else:
                        origin = r
                    if origin not in existing_nonterminals and origin not in skipped_nonterminal_names:
                        existing_nonterminals.append(origin)
                        existing_nonterminals = find_child_nonterminals(origin, ruleset, existing_nonterminals)
    return existing_nonterminals


def obtain_rule_correction_pairs(prediction, ruleset, raw_ruleset, skip_undefined_nt=True, complement_mode=False):
    """
    Returns a list of candidates in the form of (prefix, suffix).
    """  
    pred_lark = bnf2lark(prediction)
    pred_lark = dedupulicate_rules(pred_lark)
    pred_lark = ebnflark2bnflark(pred_lark)
    
    # Find valid rules
    pred_rulelist = []
    pred_lhs_rhs_dict = collections.defaultdict(list)
    for raw_rule_str in pred_lark.split("\n"):
        if ":" not in raw_rule_str:
            continue
        lhs, rhs_str = split_rule(raw_rule_str)
        rhs_options = get_rhs_options(rhs_str)
        for rhs_tokens in rhs_options:
            raw_rule = SimpleRule(lhs, tuple(rhs_tokens))
            short_rhs_tokens = get_short_tokens(rhs_tokens)
            short_rule = SimpleRule(lhs, tuple(short_rhs_tokens))
            pred_rulelist.append(short_rule)
            if lhs in skipped_nonterminal_names:
                pred_lhs_rhs_dict[lhs].append((short_rule, raw_rule))
            if short_rule in ruleset or raw_rule in ruleset:
                pred_lhs_rhs_dict[lhs].append((short_rule, raw_rule))
    
    # Remove rules with loops and find nonterminals and terminals in the valid rules
    valid_rules, defined_nonterminals = [], []
    nonterminals_in_valid_rules, terminals_in_valid_rules = [], []
    for lhs, rhs_info in pred_lhs_rhs_dict.items():
        non_loop_rhs_info = []
        for short_rule, raw_rule in rhs_info:
            if not detect_loop(lhs, short_rule.expansion):
                non_loop_rhs_info.append((short_rule, raw_rule))
        for short_rule, raw_rule in non_loop_rhs_info:
            valid_rules.append(raw_rule)
            defined_nonterminals.append(lhs)
            for r in short_rule.expansion:
                if re.match(r'^".*"$', r):
                    terminals_in_valid_rules.append(r)
                else:
                    if r[-1] in ["+", "?"]:
                        nonterminals_in_valid_rules.append(r[:-1])
                    else:
                        nonterminals_in_valid_rules.append(r)
    
    # Find unused nonterminals in the rhs of the valid rules
    unused_nonterminals = set(defined_nonterminals) - set(nonterminals_in_valid_rules) - set(["game"])
    # Find undefined nonterminals in the rhs of the valid rules
    if skip_undefined_nt:
        undefined_nonterminals = list(set(nonterminals_in_valid_rules) - set(defined_nonterminals) - set(skipped_nonterminal_names))
    else:
        undefined_nonterminals = list(set(nonterminals_in_valid_rules) - set(defined_nonterminals))

    # Construct the prefix
    unique_valid_rules = []
    for rule in valid_rules:
        if rule not in unique_valid_rules and rule.origin not in unused_nonterminals:
            unique_valid_rules.append(rule)
    prefix = rulelist2bnfstr(unique_valid_rules)
    
    if complement_mode:
        child_undefined_nonterminals = copy.deepcopy(undefined_nonterminals)
        for undefined_nonterminal in undefined_nonterminals:
            child_undefined_nonterminals = find_child_nonterminals(undefined_nonterminal, ruleset, child_undefined_nonterminals)
        if skip_undefined_nt:
            undefined_nonterminals = list(set(child_undefined_nonterminals + undefined_nonterminals) - set(defined_nonterminals) - set(skipped_nonterminal_names))
        else:
            undefined_nonterminals = list(set(child_undefined_nonterminals + undefined_nonterminals) - set(defined_nonterminals))
        
    if len(undefined_nonterminals) == 0:
        return prefix, []
    
    gtrules_for_udnonterms = collections.defaultdict(list)
    for r in raw_ruleset:
        if r.origin in undefined_nonterminals:
            gtrules_for_udnonterms[r.origin].append(r)
    suffix = []
    for nonterm, rules in gtrules_for_udnonterms.items():
        if len(rules) == 1 and len(rules[0].expansion) == 1:
            prefix += "\n" + rulelist2bnfstr(rules[0:1])
        else:
            suffix.append(rulelist2bnfstr(rules))
    
    return prefix, suffix


def post_process_prediction(prediction):
    pred_lark = bnf2lark(prediction)
    pred_lark = ebnflark2bnflark(pred_lark)
    rulelist = []
    is_include_int_constant, is_include_float_constant, is_include_boolean_constant, is_include_number_constant \
        = False, False, False, False
    defined_nonterminals = []
    for rule in larkstr2rulelist(pred_lark):
        if rule.origin == "int" and "INT_CONSTANT" in rule.expansion:
            is_include_int_constant = True
        elif rule.origin == "float" and "FLOAT_CONSTANT" in rule.expansion:
            is_include_float_constant = True
        elif rule.origin == "bool" and "BOOLEAN_CONSTANT" in rule.expansion:
            is_include_boolean_constant = True
        elif rule.origin == "number" and "NUMBER_CONSTANT" in rule.expansion:
            is_include_number_constant = True
        defined_nonterminals.append(rule.origin)
        rulelist.append(rule)
    
    if "int" in defined_nonterminals and not is_include_int_constant:
        rulelist.append(SimpleRule("int", ["INT_CONSTANT"]))
    if "float" in defined_nonterminals and not is_include_float_constant:
        rulelist.append(SimpleRule("float", ["FLOAT_CONSTANT"]))
    if "bool" in defined_nonterminals and not is_include_boolean_constant:
        rulelist.append(SimpleRule("bool", ["BOOLEAN_CONSTANT"]))
    if "number" in defined_nonterminals and not is_include_number_constant:
        rulelist.append(SimpleRule("number", ["NUMBER_CONSTANT"]))
    
    return rulelist2bnfstr(rulelist)


def decode_rule(llm, prompt_template, rule_user_prompt_dict, ruleset, raw_ruleset, 
                constraint_mode=False, max_num_correction=20, num_shot=3):
    rules_by_origin = collections.defaultdict(list)
    for rule in ruleset:
        rules_by_origin[rule.origin].append(rule)
    
    llm_call_count = 0
    records: list[DecodingRecord] = []
    
    # initial prediction
    if num_shot == 0:
        init_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_grammar_generation"]
    else:
        init_sys_prompt = prompt_template["system_prompt_template"]["grammar_generation"]
    init_user_prompt = dict_to_xml(rule_user_prompt_dict)
    init_counter = collections.Counter()
    init_counter_key = init_sys_prompt + init_user_prompt
    bnf_xml_pattern = re.compile(r'<bnf_grammar_rules>(.*?)</bnf_grammar_rules>', re.DOTALL)
    rule_prediction, response_text, init_counter, init_llm_call_count = llm_iterative_completion(
        llm, init_sys_prompt, init_user_prompt, init_counter, init_counter_key, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION, bnf_xml_pattern)
    llm_call_count += init_llm_call_count
    records.append(RuleDecodingRecord(DecodingRecord(init_sys_prompt, init_user_prompt, response_text, rule_prediction), "", []))
    logger.info("-" * 20)
    logger.info("[Rule] Initial prediction")
    logger.info(f"rule_prediction: \n{rule_prediction}")

    correction_count = 0
    counter = collections.Counter()
    is_interrupted = False
    while correction_count < max_num_correction:
        logger.debug("-" * 40)
        logger.debug(f'[Rule] Correction Count: {correction_count}')
        
        prefix, suffix_list = obtain_rule_correction_pairs(rule_prediction, ruleset, raw_ruleset, llm.args.platform!="local")
        rule_prediction = prefix
        if len(suffix_list) == 0 and len(rule_prediction) > 0:
            logger.info("*" * 20)
            logger.info("[Rule][Success] No undefined non-terminals found")
            break
        logger.debug(f"[Rule] prefix: \n{prefix}")
        logger.debug(f"[Rule] suffix: \n{suffix_list}")
        
        if constraint_mode:
            rule_prediction, local_llm_call_count, is_repetition, all_response_text, rule_sys_prompt, rule_user_prompt = constrained_decoding(
                prefix, suffix_list, llm, prompt_template, rule_user_prompt_dict, rule_prediction, counter, num_shot)
        else:
            rule_prediction, local_llm_call_count, is_repetition, all_response_text, rule_sys_prompt, rule_user_prompt = unconstrained_decoding(
                prefix, suffix_list, llm, prompt_template, rule_user_prompt_dict, rule_prediction, counter, num_shot)
        llm_call_count += local_llm_call_count
        
        if is_repetition:
            logger.info("[Rule][Interruption] Rule repetition detected")
            is_interrupted = True
            break
        
        # logger.debug(f"[Rule] rule_prediction: \n{rule_prediction}")
        records.append(
            RuleDecodingRecord(
                DecodingRecord(rule_sys_prompt, rule_user_prompt, all_response_text, rule_prediction), 
                prefix, 
                suffix_list
                )
            )
        
        correction_count += 1
    
    complement_suffix = ""
    if correction_count == max_num_correction or is_interrupted:
        logger.info("[Rule] Complement mode for undefined non-terminals")
        prefix, complement_suffix = obtain_rule_correction_pairs(rule_prediction, ruleset, raw_ruleset, llm.args.platform!="local", complement_mode=True)
        complement_suffix = "\n".join(complement_suffix).strip()
        rule_prediction = prefix
    
    post_processed_rule_prediction = post_process_prediction(rule_prediction)
    if llm.get_input_ids("", post_processed_rule_prediction).shape[1] < (MAX_TOKEN_LEN // 2):
        logger.info("[Rule] Post-processed rule prediction")
        rule_prediction = post_processed_rule_prediction

    return rule_prediction, llm_call_count, complement_suffix


def escape_special_chars(match):
    s = match.group(0)
    escaped = re.sub(r'([.^$*+?{}[\]|()\\])', r'\\\1', s)
    return escaped

def escape_within_quotes(text):
    pattern = re.compile(r'"([^"]*)"')
    result = pattern.sub(escape_special_chars, text)
    return result

def add_brackets(clause):
    if clause[-1] in ["+", "?"]:
        if clause[-2] == ")":
            return f"{add_brackets(clause[:-2])}){clause[-1]}"
        else:
            return f"({add_brackets(clause[:-1])}){clause[-1]}"
    else:
        return clause

def larkrhs2regex(rhs):
    new_rhs = []
    for r in rhs.split():
        new_r = add_brackets(r)
        new_r = escape_within_quotes(new_r)
        new_rhs.append(new_r)
    return " ".join(new_rhs)

def extract_rhs_list(grammar_rhs: str):
    rhs_list = []
    current_segment = ""
    bracket_num = 0
    for char in grammar_rhs:
        if char == '|':
            if current_segment and bracket_num == 0:
                rhs_list.append(current_segment.strip())
                current_segment = ""
            else:
                current_segment += char
        elif char == '(':
            bracket_num += 1
            current_segment += char
        elif char == ')':
            bracket_num -= 1
            current_segment += char
        else:
            current_segment += char
    if current_segment:
        rhs_list.append(current_segment.strip())
    return rhs_list


def unconstrained_decoding(prefix, suffix_list, llm, prompt_template, rule_user_prompt_dict, rule_prediction, counter, num_shot):
    remaining_bnf_xml_pattern = re.compile(r'<remaining_bnf_grammar_rules>(.*?)</remaining_bnf_grammar_rules>', re.DOTALL)
    llm_call_count = 0
    if num_shot == 0:
        rule_corr_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_grammar_correction"]
    else:
        rule_corr_sys_prompt = prompt_template["system_prompt_template"]["grammar_correction"]
    all_response_text = ""
    is_repetition = False
    rule_user_prompt = ""
    for suffix in suffix_list:
        undefined_nonterminal, rhs = suffix.split("::=")
        correct_rule_user_prompt_dict = copy.deepcopy(rule_user_prompt_dict)
        correct_rule_user_prompt_dict["task"]["valid_bnf_grammar_rules"] = dedent(prefix)
        correct_rule_user_prompt_dict["task"]["undefined_nonterminal"] = dedent(undefined_nonterminal.strip())
        correct_rule_user_prompt_dict["task"]["reference_grammar_rules"] = dedent(suffix)
        rule_user_prompt = dict_to_xml(correct_rule_user_prompt_dict)
        rule_user_prompt += prompt_template["grammar_correction_condition"]
        counter_key = prefix + suffix
        partial_rule_prediction, response_text, counter, local_llm_call_count = llm_iterative_completion(
            llm, rule_corr_sys_prompt, rule_user_prompt, counter, counter_key, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION, remaining_bnf_xml_pattern)
        if counter[counter_key] > MAX_REPETITION:
            is_repetition = True
            break
        rule_prediction += "\n" + partial_rule_prediction
        llm_call_count += local_llm_call_count
        all_response_text += "\n" + response_text
    return rule_prediction, llm_call_count, is_repetition, all_response_text, rule_corr_sys_prompt, rule_user_prompt
    

def constrained_decoding(prefix, suffix_list, llm, prompt_template, rule_user_prompt_dict, rule_prediction, counter, num_shot):
    llm_call_count = 0
    if num_shot == 0:
        rule_corr_sys_prompt = prompt_template["system_prompt_template"]["zero-shot_grammar_correction_choice"]
    else:
        rule_corr_sys_prompt = prompt_template["system_prompt_template"]["grammar_correction_choice"]
    all_response_text = ""
    is_repetition = False
    rule_user_prompt = ""
    for suffix in suffix_list:
        correct_rule_user_prompt_dict = copy.deepcopy(rule_user_prompt_dict)
        correct_rule_user_prompt_dict["task"]["reference_grammar_rules"] = dedent(suffix)
        correct_rule_user_prompt_dict["task"]["bnf_grammar_rules"] = dedent(prefix)
        rule_user_prompt = dict_to_xml(correct_rule_user_prompt_dict)
        rule_user_prompt = rule_user_prompt.replace("</bnf_grammar_rules>\n</task>", "")
        counter_key = prefix + suffix
        undefined_nonterminal, rhs = suffix.split(" ::= ")
        _rule_user_prompt = rule_user_prompt.strip() + "\n" + undefined_nonterminal + " ::= "
        rhs_choices = extract_rhs_list(rhs)
        choices = [larkrhs2regex(choice) for choice in rhs_choices]
        
        response_text, counter, local_llm_call_count = llm_iterative_choice(
            llm, rule_corr_sys_prompt, _rule_user_prompt, choices, counter, counter_key, MAX_LOCAL_NUM_CORRECTION, MAX_REPETITION)
        partial_rule_prediction = undefined_nonterminal + " ::= " + re.sub(r'\s+', ' ', response_text)
        
        if counter[counter_key] > MAX_REPETITION:
            is_repetition = True
            break
        rule_prediction += "\n" + partial_rule_prediction
        llm_call_count += local_llm_call_count
        all_response_text += "\n" + response_text
    return rule_prediction, llm_call_count, is_repetition, all_response_text, rule_corr_sys_prompt, rule_user_prompt
