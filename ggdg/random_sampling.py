import random
import string

from minEarley.parser import EarleyParser
from ggdg.utils import *
from ggdg.game_description_decoding import obtain_program_correction_pairs


RANDOM_MAX_STRING_LENGTH = 255
RANDOM_MIN_STRING_LENGTH = 1
RANDOM_MAX_FLOAT = 10000000.0
RANDOM_MIN_FLOAT = -11.59
RANDOM_MAX_INT = 10000000
RANDOM_MIN_INT = -216


def generate_random_string(min_length, max_length):
    length = random.randint(min_length, max_length)
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def generate_random_float(min_value, max_value):
    return random.uniform(min_value, max_value)


def generate_random_int(min_value, max_value):
    return random.randint(min_value, max_value)


def randomly_decode_game_description(lark_grammar_for_local, use_oracle_rule_flag, global_parser):
    try:
        if use_oracle_rule_flag:
            # lark_grammar_for_local = lark_grammar_for_local.replace("INT_CONSTANT", "int_constant")
            # lark_grammar_for_local = lark_grammar_for_local + '\n%import common.WS\n%ignore WS'
            parser = EarleyParser(decorate_oracle_grammar_ludii(lark_grammar_for_local), start=global_parser.option.start, oracle_rule=True)
        else:
            parser = EarleyParser(decorate_grammar_ludii(lark_grammar_for_local, replace_prefix=False), start=global_parser.option.start)
        logger.info(f"local parser created with predicted grammar!!!!!!!")
    except Exception as e:
        logger.warning(f"failed to create parser due to {e}, reverting to global parser")
        return ""

    sampling_count = 0
    program = "(game"
    pairs = obtain_program_correction_pairs(program, parser)
    while len(pairs) > 0 and sampling_count < 100:
        prefix, suffix_list = pairs
        next_suffix = random.choice(list(suffix_list))
        if next_suffix == "STRING":
            next_suffix = '"' + generate_random_string(RANDOM_MIN_STRING_LENGTH, RANDOM_MAX_STRING_LENGTH) + '"'
        elif next_suffix == "INT_CONSTANT":
            next_suffix = str(generate_random_int(RANDOM_MIN_INT, RANDOM_MAX_INT))
        elif next_suffix == "FLOAT_CONSTANT":
            next_suffix = str(generate_random_float(RANDOM_MIN_FLOAT, RANDOM_MAX_FLOAT))
        program = prefix + " " + next_suffix
        pairs = obtain_program_correction_pairs(program, parser)
        sampling_count += 1
    return program
