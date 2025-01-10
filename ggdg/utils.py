import collections
import re
from math import isnan

import pandas as pd
from lark.load_grammar import _TERMINAL_NAMES, load_grammar
from lark.grammar import Terminal
from dataclasses import dataclass

from minEarley.tree import Tree
from ggdg.train_utils import logger

"""
For convenince, we use SimpleRule instead of lark.grammar.Rule for 1) putting rules 
in the instruction, 2) check if model-generated rules are valid. 
In the future, we may want to directly use lark.grammar.Rule, e.g., let the model
generate rules in EBNF or BNF format.
"""

# these nonterminals will be inlined when constructing rules
inline_terminal_names = {
        # for SMC dataset
        "WORD", "NUMBER", "ESCAPED_STRING", "L", 
        # for regex dataset
        "STRING", "INT", "CHARACTER_CLASS", "CONST",
        # for overnight
        # "PROPERTY", "SINGLETON_VALUE", "ENTITY_VALUE", "NUMBER_VALUE",
        # for molecule
        "N", "C", "O", "F", "c",
        # for fol
        "PNAME", "CNAME", "LCASE_LETTER"
}
for k, v in _TERMINAL_NAMES.items():
    inline_terminal_names.add(v)

## these are the nonterminals that are not needed to be predicted from model, will be used to to check the validity of the generated rules
skipped_nonterminal_names = (
    # for smc and regex
    "string", "number", "literal", "delimiter", 
    "int_constant", "float_constant", "boolean_constant", "number_constant",
    "INT_CONSTANT", "FLOAT_CONSTANT", "BOOLEAN_CONSTANT", "NUMBER_CONSTANT",
)

"""
Some concepts:
    - larkstr: a string in Lark format 
    - bnfstr: a string in BNF format (use ::= instead of :)
"""

special_terminals = [
    "INT_CONSTANT", "FLOAT_CONSTANT", "STRING"
]


# poor man's rule
@dataclass
class SimpleRule:
    origin: str
    expansion: tuple

    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return self.to_lark()
    
    def to_lark(self):
        return f"{self.origin} : {' '.join(self.expansion)}"
    
    def to_bnf(self):
        return f"{self.origin} ::= {' '.join(self.expansion)}"
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SimpleRule):
            return False
        return str(self).replace("?", "") == str(__o).replace("?", "")

def _wrap_string(s):
    if s.startswith("\"") and s.endswith("\""):
        # a bit complex to preserve the quotation marks
        s = f"\"\\{s[:-1]}\\\"{s[-1]}"
    else:
        s = f"\"{s}\""    

    # escape unicode characters 
    if "\\u" in s:
        s = s.replace("\\u", "\\\\u")
    
    return s

def split_rule(rule):
    split_idx = rule.index(":")
    lhs, rhs = rule[:split_idx].strip(), rule[split_idx+1:].strip()
    return lhs, rhs

def treenode2rule(treenode):
    if treenode is None:
        return None

    if isinstance(treenode, Tree):
        origin = f"{treenode.data.value}"
        expansion = []

        for child in treenode.children:
            if child is None:
                continue

            if isinstance(child, Tree):
                expansion.append(child.data.value)
            else:
                if child.type.startswith("__") or child.type in inline_terminal_names:
                    expansion.append(_wrap_string(child.value))
                else:
                    expansion.append(child.type)
    else: # terminal
        if treenode.type.startswith("__") or treenode.type in inline_terminal_names:
            return None
        else:
            origin = treenode.type
            expansion = [_wrap_string(treenode.value)]
    return SimpleRule(origin, tuple(expansion))
    
def extract_rule_stat(tree, rule_stat):
    """
    Count the occurrence of each rule
    """
    cur_rule = treenode2rule(tree)
    if cur_rule is None:
        return
    if cur_rule not in rule_stat:
        rule_stat[cur_rule] = 1
    else:
        rule_stat[cur_rule] += 1

    if getattr(tree, "children", None):
        for child in tree.children:
            extract_rule_stat(child, rule_stat)

def tree2rulelist(tree):
    rule_list = []
    def recur_add(node, rule_list):
        cur_rule = treenode2rule(node)
        if cur_rule is None:
            return
        rule_list.append(cur_rule)

        if getattr(node, "children", None):
            for child in node.children:
                recur_add(child, rule_list)
    recur_add(tree, rule_list)
    return rule_list

def linearize_tree(tree):
    def recur_add(node):
        if getattr(node, "children", None) is None:
            return "{" + f"{node.value}" + "}"
        else:
            ret_str = f"[{node.data.value} "
            for child in node.children:
                ret_str += recur_add(child)
                ret_str += " "
            ret_str += "]"
            return ret_str
    return recur_add(tree)

def linearized_tree_to_program(linearized_tree, delimiter=""):
    tokens = re.findall(r'{(.*?)}', linearized_tree)
    return delimiter.join(tokens)

def normalize_program(program, parser):
    tree = parser.parse(program)
    linearized_tree = linearize_tree(tree)
    return linearized_tree_to_program(linearized_tree)

def rulelist2larkstr(rule_stat):
    lhs2rhs = collections.OrderedDict()
    for rule in rule_stat:
        lhs, rhs = rule.origin, rule.expansion
        if lhs not in lhs2rhs:
            lhs2rhs[lhs] = []
        lhs2rhs[lhs].append(rhs)
    
    grammar = ""
    for lhs in lhs2rhs:
        grammar += f"{lhs} :"
        for rhs in lhs2rhs[lhs]:
            rhs_str = " ".join(rhs)
            grammar += f" {rhs_str} |"
        grammar = grammar[:-2]
        grammar += "\n"
    
    return grammar.strip()

def rulelist2bnfstr(rule_list):
    """
    Convert list of rules to lark grammar string
    """
    larkstr = rulelist2larkstr(rule_list)
    bnf_str = lark2bnf(larkstr)
    return bnf_str

def extract_min_grammar_from_trees(trees, return_rules=False):
    """
    Extract minimal grammar to reconstruct the tree
    """
    rule_stat = collections.OrderedDict()
    for tree in trees:
        extract_rule_stat(tree, rule_stat)
    grammar = rulelist2larkstr(rule_stat)

    if return_rules:
        return grammar, list(rule_stat.keys())
    else:
        return grammar

def lark2bnf(grammar):
    """
    Make it easier for GPT to generate
    """
    #grammar = grammar.replace(" : ", " -> ")
    grammar = grammar.replace(" : ", " ::= ")
    return grammar

def bnf2lark(grammar):
    """
    Opposite of lark2bnf 
    """
    # grammar = grammar.replace(" -> ", " : ")
    grammar = grammar.replace(" ::= ", " : ")
    return grammar

def get_lhs_raw_rhs_dict(raw_str):
    pred_lhs_raw_rhs_dict = {}
    for raw_line in raw_str.split("\n"):
        raw_line = raw_line.strip()
        lhs, raw_rhs = raw_line.split("::=")
        lhs = lhs.strip()
        raw_rhs = raw_rhs.strip()
        pred_lhs_raw_rhs_dict[lhs] = SimpleRule(lhs, tuple(raw_rhs.split()))
    return pred_lhs_raw_rhs_dict

def decorate_grammar(grammar):
    """
    Add auxiliary rules to the grammar
    """
    grammar += "\n%import common.DIGIT"
    grammar += "\n%import common.LCASE_LETTER"
    grammar += "\n%import common.UCASE_LETTER"
    grammar += "\n%import common.WS"
    grammar += "\n%ignore WS"
    return grammar

# expression for lark
PREFIX_DICT = {
    "NUMBER_CONSTANT": "NUMBER_CONSTANT : FLOAT_CONSTANT | INT_CONSTANT",
    "FLOAT_CONSTANT": 'FLOAT_CONSTANT : /-?\d+(\.\d+)?/ | /-?\.\d+/',
    "INT_CONSTANT": 'INT_CONSTANT : /-?\d+/',
    "BOOLEAN_CONSTANT": 'BOOLEAN_CONSTANT : "True" | "False"',
    "string": "string : string | STRING",
    "STRING": 'STRING : /"([^"\\\\]|\\\\.)*"/',
}

def decorate_grammar_ludii(grammar, replace_prefix=True):
    if len(grammar) == 0:
        return grammar
    use_rules, special_rules = [], []
    for rule in grammar.split("\n"):
        lhs, rhs = split_rule(rule)
        if replace_prefix:
            if lhs not in PREFIX_DICT:
                use_rules.append(rule)
        else:
            if lhs in PREFIX_DICT:
                special_rules.append(lhs)
            use_rules.append(rule)
    grammar = "\n".join(use_rules)
    for prefix_key, prefix_rule in PREFIX_DICT.items():
        if prefix_key not in special_rules and prefix_key in grammar:
            grammar = grammar + "\n" + prefix_rule
    grammar += "\n%import common.WS\n%ignore WS"
    return grammar

def decorate_oracle_grammar_ludii(grammar):
    PREFIX = ['%import common.WS',
              '%ignore WS',]
    grammar += "\n" + "\n".join(PREFIX)
    return grammar

def collect_rules_from_examples(programs, parser):
    """
    Parse programs to extract rules and collect them. Mostly for debugging
    """
    rule_stat = collections.OrderedDict()
    for program in programs:
        tree = parser.parse(program)
        extract_rule_stat(tree, rule_stat)
    
    rulestr_set = set()
    for rule in rule_stat:
        rulestr = str(rule).strip()
        rulestr_set.add(rulestr)
    return rulestr_set

def collect_rules_from_larkfile(lark_file):
    """
    Parse bnf file (.lark) to extract rules
    """
    rule_stat = collections.OrderedDict() # used as ordered set

    with open(lark_file, "r") as f:
        cur_nonterminal, cur_rhs = None, None
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("//"):
                continue
            elif line.startswith("|"):
                cur_rhs += " " + line.strip()
            elif ":" in line and "\":" not in line: # for rules like :duration
                if cur_nonterminal is not None and cur_rhs is not None:
                    rhs_list = []
                    count_bracket = 0
                    for cur_rhs_part in cur_rhs.split():
                        if cur_rhs_part == "|" and count_bracket == 0:
                            rule = SimpleRule(cur_nonterminal, tuple(rhs_list))
                            rule_stat[rule] = 1
                            rhs_list = []
                        else:
                            rhs_list.append(cur_rhs_part)
                        if "(" in cur_rhs_part:
                            count_bracket += 1
                        if ")" in cur_rhs_part:
                            count_bracket -= 1
                    if len(rhs_list) > 0:
                        rule = SimpleRule(cur_nonterminal, tuple(rhs_list))
                        rule_stat[rule] = 1
                lhs, rhs = split_rule(line)
                cur_nonterminal = lhs
                cur_rhs = rhs.strip()
            elif line.startswith("%"):
                continue
            else:
                raise ValueError(f"Unknown line: {line}")
    rule_set = list(rule_stat.keys())
    return rule_set, None


def concatenate_consecutive_terminals(token_list):
    new_token_list = []
    for i, token in enumerate(token_list):
        if token.startswith('"') and token.endswith('"') and token not in ['"{"', '"}"']:
            if i > 0 and new_token_list[-1].startswith('"') and new_token_list[-1].endswith('"') and new_token_list[-1] not in ['"{"', '"}"']:
                new_token_list[-1] = new_token_list[-1][:-1]
                token = token[1:]
        new_token_list.append(token)
    return new_token_list


def collect_rules_from_parser(parser):
    rule_list = []
    nonterminals_by_name = {}
    for rule in parser.rules:
        if "__" in rule.origin.name and rule.origin.name not in nonterminals_by_name:
            nonterminals_by_name[rule.origin.name] = rule.expansion[0].name
    
    terminals = []
    for rule in parser.rules:
        if type(rule.origin.name) == str:
            origin = rule.origin.name
        else:
            origin = rule.origin.name.value
        if "__" in origin:
            continue
        expansion = rule.expansion
        processed_expansion = []
        include_plus = False
        for i, e in enumerate(expansion):
            if isinstance(e, Terminal):
                if e.name in special_terminals:
                    raw_e = e.name
                else:
                    raw_e = parser.lexer_conf.terminals_by_name[e.name].pattern.raw
                if raw_e is None:
                    continue
                if raw_e[1:-1].lower() == e.name.lower():
                    processed_expansion.append(e.name)
                    if raw_e not in terminals:
                        terminals.append(raw_e) # terminalを定義する規則はparser.rulesに含まれないので別途追加
                else:
                    processed_expansion.append(raw_e)
            else:
                if e.name in nonterminals_by_name:
                    if nonterminals_by_name[e.name] in parser.lexer_conf.terminals_by_name.keys():
                        e_name = parser.lexer_conf.terminals_by_name[nonterminals_by_name[e.name]].pattern.raw
                    else:
                        e_name = nonterminals_by_name[e.name]
                    if "plus" in e.name:
                        processed_expansion.append(f"{e_name}+")
                        include_plus = True
                    else:
                        processed_expansion.append(e_name)
                else:
                    processed_expansion.append(e.name)

        if include_plus:
            processed_expansion_non_plus = [p_e.replace("+", "") for p_e in processed_expansion]
            rule_list.append(SimpleRule(origin, tuple(processed_expansion_non_plus)))
        
        if len(processed_expansion) == 2 and processed_expansion[-1] == '")"':
            processed_expansion = [processed_expansion[0][:-1] + processed_expansion[-1][1:]]
            
        rule_list.append(SimpleRule(origin, tuple(processed_expansion)))
        
        # '"(dominoes"', '"upTo:"', 'int', '")"' -> '"(dominoes', 'upTo:"', 'int', '")"'
        additional_processed_expansion = concatenate_consecutive_terminals(processed_expansion)
        if additional_processed_expansion != processed_expansion:
            rule_list.append(SimpleRule(origin, tuple(additional_processed_expansion)))
    
    for terminal in terminals:
        rule_list.append(SimpleRule(terminal[1:-1].upper(), tuple([terminal])))
    return rule_list

def str2rulestr(string):
    string = string.strip()
    rule_str_list = []
    temp_rule_str = ""
    is_next_add = False
    for line in string.split("\n"):
        line = line.strip()
        if " ::= " in line:
            rule_str_list.append(temp_rule_str.strip())
            temp_rule_str = line
        elif line.startswith("|") or is_next_add:
            temp_rule_str += " " + line.strip()
            is_next_add = False
        if line.endswith("|"):
            is_next_add = True
    rule_str_list.append(temp_rule_str.strip())
    return "\n".join(rule_str_list)

def flatten_choices(ebnf_rule):
    pattern = re.compile(r'\(([^()]+)\)')
    match = pattern.search(ebnf_rule)
    if not match:
        return [ebnf_rule]
    choices = match.group(1).split('|')
    start, end = match.span()
    
    combinations = [ebnf_rule[:start] + choice.strip() + ebnf_rule[end:] for choice in choices]
    
    out_rules = []
    for combination in combinations:
        out_rules += flatten_choices(combination)
    
    return out_rules

def get_all_combination_of_rhs(rhs):
    rhs = rhs.replace('"(', "#").replace('")"', "~")
    bnf_rhs_list = flatten_choices(rhs)
    bnf_rhs_list = [bnf_rhs.replace("#", '"(').replace("~", '")"') for bnf_rhs in bnf_rhs_list]
    return bnf_rhs_list

def summarize_repetition(ebnf_rule):
    ebnf_rule = re.sub(r'\?{2,}', '?', ebnf_rule)
    ebnf_rule = re.sub(r'\+{2,}', '+', ebnf_rule)
    ebnf_rule = re.sub(r'\?(\+|\+?)|\+(\?|\?+)', '?', ebnf_rule)
    return ebnf_rule

def ebnflark2bnflark(grammar_str):
    all_bnf_rules = []
    for ebnf_rule in grammar_str.split("\n"):
        if ":" not in ebnf_rule:
            continue
        lhs, rhs = split_rule(ebnf_rule)
        bnf_rhs_list = get_all_combination_of_rhs(rhs)
        bnf_rhs_list = [summarize_repetition(bnf_rhs) for bnf_rhs in bnf_rhs_list]
        all_bnf_rules += [f"{lhs} : {bnf_rhs}" for bnf_rhs in bnf_rhs_list]
    return "\n".join(all_bnf_rules)

def dedupulicate_rules(grammar_str):
    all_bnf_rules = {}
    for ebnf_rule in grammar_str.split("\n"):
        if ":" not in ebnf_rule:
            continue
        lhs, rhs = split_rule(ebnf_rule)
        if lhs not in all_bnf_rules:
            all_bnf_rules[lhs] = rhs
        elif rhs != all_bnf_rules[lhs]:
            all_bnf_rules[lhs] = " | ".join([all_bnf_rules[lhs], rhs])
    return "\n".join([f"{lhs} : {rhs}" for lhs, rhs in all_bnf_rules.items()])

def remove_redundant_curly_brackets(rhs_list):
    new_rhs_list = []
    tokens_in_brackets = 0
    in_brackets = False
    for rhs in rhs_list:
        if rhs == '"{"':
            in_brackets = True
        elif rhs == '"}"' and in_brackets:
            in_brackets = False
            if tokens_in_brackets == 1:
                new_rhs_list.pop(-2)
                continue
        elif in_brackets:
            tokens_in_brackets += 1
        new_rhs_list.append(rhs)
    return new_rhs_list

def get_rhs_options(rhs):
    rhs_options = []
    rhs_option_tokens = []
    count_bracket = 0
    for rhs_token in rhs.split():
        if rhs_token == "|" and count_bracket == 0:
            rhs_options.append(rhs_option_tokens)
            rhs_option_tokens = []
        else:
            rhs_option_tokens.append(rhs_token)
        if "(" in rhs_token:
            count_bracket += 1
        if ")" in rhs_token:
            count_bracket -= 1
    if len(rhs_option_tokens) > 0:
        rhs_options.append(rhs_option_tokens)
    return rhs_options

def larkstr2rulelist(lark_str):
    """
    Convert lark grammar string to list of rules.
    TODO: use load_grammar function from lark
    """
    for raw_rule in lark_str.split("\n"):
        if ":" not in raw_rule:
            continue
        lhs, rhs_str = split_rule(raw_rule)
        rhs_options = get_rhs_options(rhs_str)
        for rhs_tokens in rhs_options:
            new_rhs = []
            op = ""
            if len(rhs_tokens) > 1 and all(t.startswith('"') and t.endswith('"') for t in rhs_tokens):
                # すべての要素が文字列で囲まれている場合は結合
                new_rhs = ['"' + "".join([r[1:-1] for r in rhs_tokens]) + '"']
            else:
                rhs_tokens = remove_redundant_curly_brackets(rhs_tokens)
                num_rhs = len(rhs_tokens)
                for i in range(num_rhs):
                    if rhs_tokens[i][-1] in ["+", "?"]:
                        op = rhs_tokens[i][-1]
                        rhs_tokens[i] = rhs_tokens[i][:-1]
                    else:
                        op = ""
                    if i == 0 or rhs_tokens[i] != rhs_tokens[i-1]:
                        new_rhs.append(rhs_tokens[i] + op)
                    elif "+" not in new_rhs[-1]:
                        if new_rhs[-1][-1] == "?":
                            if op == "+":
                                new_rhs[-1] = new_rhs[-1][:-1] + "+"
                        else:
                            new_rhs[-1] += "+"
            rule = SimpleRule(lhs, new_rhs)
            yield rule

def get_short_tokens(rhs_tokens: list) -> list:
    new_rhs = []
    op = ""
    if len(rhs_tokens) > 1 and all(t.startswith('"') and t.endswith('"') for t in rhs_tokens):
        # すべての要素が文字列で囲まれている場合は結合
        new_rhs = ['"' + "".join([r[1:-1] for r in rhs_tokens]) + '"']
    else:
        rhs_tokens = remove_redundant_curly_brackets(rhs_tokens)
        num_rhs = len(rhs_tokens)
        for i in range(num_rhs):
            if rhs_tokens[i][-1] in ["+", "?"]:
                op = rhs_tokens[i][-1]
                rhs_tokens[i] = rhs_tokens[i][:-1]
            else:
                op = ""
            if i == 0 or rhs_tokens[i] != rhs_tokens[i-1]:
                new_rhs.append(rhs_tokens[i] + op)
            elif "+" not in new_rhs[-1]:
                if new_rhs[-1][-1] == "?":
                    if op == "+":
                        new_rhs[-1] = new_rhs[-1][:-1] + "+"
                else:
                    new_rhs[-1] += "+"
    return new_rhs  

def check_grammar_validity(valid_rules, pred_lark_str):
    """
    Check if the grammar (i.e., bnf_str produced by model) is valid
    """
    for rule in larkstr2rulelist(pred_lark_str):
        if rule.origin not in skipped_nonterminal_names and rule not in valid_rules:
            logger.debug(f"Found invalid rule {rule}")
            return False
    return True

def check_grammar_correctness(tgt_rules, pred_lark_str, debug=False):
    """
    Evaluate the correctness of the grammar
    """
    if pred_lark_str is None:
        return False
    tgt_ruleset = set(tgt_rules)
    pred_ruleset = set(larkstr2rulelist(pred_lark_str))

    if debug:
        logger.debug(f"Rules in pred but not in tgt: {pred_ruleset - tgt_ruleset}")
        logger.debug(f"Rules in tgt but not in pred: {tgt_ruleset - pred_ruleset}")

    return pred_ruleset == tgt_ruleset

def gen_min_lark(program, parser):
    """
    Obtain the minimal grammar from a program
    """
    parse_trees = []
    if "\n" in program:
        program = program.split("\n")
        for line in program:
            parse_tree = parser.parse(line)
            parse_trees.append(parse_tree)
    else:
        parse_tree = parser.parse(program)
        parse_trees.append(parse_tree)
    grammar = extract_min_grammar_from_trees(parse_trees)
    return grammar

def gen_ludii_min_lark(program, parser):
    parse_trees = []
    program = remove_newlines_in_parentheses(program)
    parse_tree = parser.parse(program)
    parse_trees.append(parse_tree)
    grammar = extract_min_grammar_from_trees(parse_trees)
    return grammar

def replace_colon_content(file_content):
    # 括弧内部と括弧外部の両方にマッチするように正規表現を調整する
    updated_content = file_content
    # 括弧の中身を処理
    bracket_pattern = re.compile(r'\b\w+:\(([^)]+)\)')
    non_bracket_pattern = re.compile(r'\b\w+:([^\s]+)')

    while True:
        # 括弧内部の置換
        if not bracket_pattern.findall(updated_content) and not non_bracket_pattern.findall(updated_content):
            break  # 置換するべき部分がなくなればループを終了
        updated_content = bracket_pattern.sub(r'(\1)', updated_content)
        # 括弧外部の置換
        updated_content = non_bracket_pattern.sub(r'\1', updated_content)
    
    return updated_content

def remove_newlines_in_parentheses(text):
    pattern = re.compile(r'\(\s*([^()\s]*?)\s*\n\s*([^()\s]*?)\s*\)', re.DOTALL)
    while True:
        new_text = pattern.sub(r'(\1\2)', text)
        if new_text == text:
            break
        text = new_text
    return new_text

def program2rules(program, parser):
    try:
        tree = parser.parse(program)
        rule_list = tree2rulelist(tree)
        return " ## ".join([rule.to_bnf() for rule in rule_list])
    except:
        # there are some bad cases, see run_parse_smc.py
        return program

def dict_to_xml(d: dict):
    xml = ""
    for k, v in d.items():
        if isinstance(v, dict):
            value_str = dict_to_xml(v)
        elif isinstance(v, list):
            value_str = list_to_xml(v, k)
        else:
            value_str = v
        xml += f"\n<{k}>\n{value_str.strip()}\n</{k}>"
    return xml.strip()


def list_to_xml(l: list, key: str):
    xml = ""
    if len(l) == 0:
        return xml
    if isinstance(l[0], dict):
        for id, item in enumerate(l):
            xml += f"\n<{key}_{id}>\n" + dict_to_xml(item) + f"\n</{key}_{id}>"
    elif isinstance(l[0], str):
        xml = ", ".join(l).strip()
    elif isinstance(l[0], list):
        for id, item in enumerate(l):
            xml += f"\n<{key}_{id}>\n" + list_to_xml(item, f"{key}_{id}") + f"\n</{key}_{id}>"
    return xml.strip()


def get_column_value(df: pd.DataFrame, column_name: str, row_id: int=0):
    if column_name in df.columns:
        if isnan(df[column_name][row_id]):
            return None
        else:
            return df[column_name][row_id]
    else:
        return None
