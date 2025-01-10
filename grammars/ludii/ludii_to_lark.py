import re
from collections import defaultdict

PREFIX = [
    'STRING: /"([^"\\\\]|\\\\.)*"/',
    '%import common.WS',
    '%ignore WS',
]

SPECIAL_NONTERMINALS = {
    'float_set_set' : ['"{"? ("{"? float+ "}"?)+ "}"?', '"{" float_set+ "}"'],
    'float_set': ['"{"? float+ "}"?', '"{" float+ "}"'],
    'int_set': ['"{"? int+ "}"?', '"{" int+ "}"'],
    'dim_set': ['"{"? dim+ "}"?', '"{" dim+ "}"'],
    'step_type_set': ['"{"? step_type+ "}"?', '"{" step_type+ "}"'],
}

origin_option_dict = defaultdict(list)

def count_leading_braces(s):
    count = 0
    start_brace = False
    for char in s:
        if not start_brace:
            if char == '{':
                start_brace = True
                count += 1
        else:
            if char == '{':
                count += 1
            else:
                break
    return count

def process_token(token: str, bracket_num: int):
    token_head, token_body, token_tail = "", token, ""
    
    if token[0] == "(" and token[-1] == ")":
        token_body = token[1:-1]
        token_head = '"("'
        token_tail = '")"'
    elif token[0] == "(":
        if any(c in token[1:] for c in ["<", "{", "[", "("]) or token[-1] == ":":
            token_head = '('
        else:
            token_head = '"('
            token_tail = '"'
        token_body = token[1:]
        bracket_num += 1
    elif token[-1] == ")":
        token_body = token[:-1]
        if bracket_num == 1:
            token_tail = '")"'
        else:
            token_tail = ')'
        bracket_num -= 1
    elif token[0] == "[" and token[-1] == "]":
        token_body = token[1:-1]
        token_tail = "?"
    elif token[0] == "[":
        token_head = '('
        token_body = token[1:]
    elif token[-1] == "]":
        token_body = token[:-1]
        token_tail = ')?'
    elif token[0] == "{" and token[-1] == "}":
        token_head = '"{"? '
        token_body = token[1:-1]
        token_tail = '+ "}"?'
    elif token[0] == "{":
        token_head = '"{"? ('
        token_body = token[1:]
    elif token[-1] == "}":
        token_body = token[:-1]
        token_tail = ')+ "}"?'
    
    if "::=" in token_body:
        token_body = re.sub(r'\s*::=', ':', token_body)
        token_body = token_body.replace("!=", "not_equal_to")
        token_body = token_body.replace("=", "equal_to")
        token_body = token_body.replace(">", "greater_than")
        token_body = token_body.replace("<", "less_than")
        token_body = token_body.replace(">=", "greater_than_or_equal_to")
        token_body = token_body.replace("<=", "less_than_or_equal_to")
        token_body = token_body.replace("%", "percent")
    elif not any(c in token_body for c in ["[", "]", "(", ")"]) and ":" in token_body:
        token_body = f'"{token_body}"'
    
    if any(c in token_body for c in ["{", "}", "[", "]", "(", ")"]):
        token_body, bracket_num = process_token(token_body, bracket_num)
    
    if (token_body[0] == "<" and token_body[-1] == ">"):
        if any(char.isupper() for char in token_body) and not any(c in token_body for c in ["NUMBER", "STRING", "FLOAT"]):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', token_body)
            token_body = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        token_body = token_body[1:-1]
    elif token_body not in ["|", ":"] and token_body[0].isupper() and not any(c in token_body for c in ["NUMBER", "STRING", "FLOAT"]):
        token_body = f'"{token_body}"'
    
    if token_body[-1] in ["+", "?"] and (len(token_tail) > 0 and token_tail[0] in ["+", "?"]):
        token_body = "(" + token_body + ")"
    
    processed_token = token_head + token_body + token_tail
    return processed_token, bracket_num

def post_process_token(token: str):
    token = token.replace("<string>", "STRING")
    token = token.replace("<int>", "NUMBER")
    token = token.replace("<float>", "FLOAT")
    token = token.replace(".", "_")
    token = token.replace("_*", "_times")
    token = token.replace("_+", "_plus")
    token = token.replace("_-", "_minus")
    token = token.replace("_/", "_div")
    token = token.replace("_^", "_pow")
    token = token.replace("swap_swap", "swap")
    if token == "!=":
        token = "not_equal_to"
    elif token == "=":
        token = "equal_to"
    elif token == ">":
        token = "greater_than"
    elif token == "<":
        token = "less_than"
    elif token == ">=":
        token = "greater_than_or_equal_to"
    elif token == "<=":
        token = "less_than_or_equal_to"
    elif token == "%":
        token = "percent"
    elif token == "(<":
        token = '"(<"'
    elif token == "(<=":
        token = '"(<="'
    return token

def extract_rhs(grammar: str):
    rhs_list = []
    current_segment = ""
    bracket_num = 0
    for char in grammar:
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

def post_process_grammar(grammar: str, terminals_dict: dict):
    grammar_lhs_str, grammar_rhs_str = grammar.split(" : ")
    grammar_lhs = grammar_lhs_str.strip()
    grammar_rhs_list = extract_rhs(grammar_rhs_str.strip())
    grammar_rhs_use_list = []
    for grammar_rhs in grammar_rhs_list:
        if grammar_rhs[:3] == '"("' and grammar_rhs[-3:] == '")"' and ' ' not in grammar_rhs[3:-3]:
            grammar_rhs = '"(' + grammar_rhs[3:-3] + ')"'
        if "board_track" in grammar_lhs:
            grammar_rhs = '"(track" string (int_set | string) ("loop:" boolean)? (int | role_type)? ("directed:" boolean)? ")"'
        if "move" in grammar_lhs and "Swap" in grammar_rhs and "Players" in grammar_rhs:
            grammar_rhs = '"(move" "Swap" "Players" (int | role_type) (int | role_type) then?")"'
        if "diagonals_type" in grammar_lhs:
            grammar_rhs = '"Alternating" \n    | "Concentric" \n    | "Implied" \n    | "Radiating" \n    | "Solid" \n    | "SolidNoSplit"'
        if grammar_rhs in grammar_rhs_use_list or 'region"' in grammar_rhs:
            continue
        
        end_pattern = re.compile(r'(\S)(\"\)\")')
        grammar_rhs = end_pattern.sub(r'\1 \2', grammar_rhs)
        
        use_grammar_rhs = []
        grammar_rhs_parts = grammar_rhs.split()
        # remove loop
        if set(grammar_rhs_parts) == set([f'"({grammar_lhs}"', grammar_lhs, '")"']) or grammar_lhs == grammar_rhs:
            continue
        for part_id, grammar_rhs_part in enumerate(grammar_rhs_parts):
            if re.match(r'^".*"$', grammar_rhs_part) and "(" not in grammar_rhs_part and ")" not in grammar_rhs_part and ":" not in grammar_rhs_part:
                key = grammar_rhs_part[1:-1].upper()
                if key not in terminals_dict:
                    terminals_dict[key] = grammar_rhs_part
                grammar_rhs_part = key
            use_grammar_rhs.append(grammar_rhs_part)
        
        grammar_rhs = " ".join(use_grammar_rhs)
            
        for special_nonterminal, rhs_info in SPECIAL_NONTERMINALS.items():
            replace_rhs, _ = rhs_info
            if ' ('+replace_rhs+')' in grammar_rhs:
                grammar_rhs = grammar_rhs.replace('('+replace_rhs+')', special_nonterminal)
            elif replace_rhs in grammar_rhs:
                grammar_rhs = grammar_rhs.replace(replace_rhs, special_nonterminal)
        
        if "times" in grammar_lhs and "*" in grammar_rhs:
            grammar_rhs_use_list.append(grammar_rhs.replace("*", "mul"))
        if "pow" in grammar_lhs and "^" in grammar_rhs:
            grammar_rhs_use_list.append(grammar_rhs.replace("^", "pow"))
        
        grammar_rhs_use_list.append(grammar_rhs)
    
    if "moves" == grammar_lhs or "effect" == grammar_lhs:
        grammar_rhs_use_list.append("seq")
    if "dim" == grammar_lhs:
        grammar_rhs_use_list.append('ints')
    
    grammar = grammar_lhs + " : " + "\n    | ".join(list(set(grammar_rhs_use_list)))
    return grammar, terminals_dict

def tokenize_option(formatted_rule: str):
    lhs, rhs = formatted_rule.split("::=")
    rhs_tokens = []
    for token in rhs.split():
        if ":" in token:
            option, token = token.split(":")
            rhs_tokens.append(option+":")
        rhs_tokens.append(token)
    new_formatted_rule = lhs + "::= " + " ".join(rhs_tokens)
    return new_formatted_rule

def process_line(formatted_rule: str, lark_grammar: list, terminals_dict: dict):
    formatted_rule = re.sub(r'(?<! )\|', ' |', formatted_rule)
    formatted_rule = re.sub(r'\|(?![ ])', '| ', formatted_rule)
    
    nested_braces_pattern = re.compile(r'\{{2}(\w+)\}{2}')
    while re.search(nested_braces_pattern, formatted_rule):
        formatted_rule = re.sub(nested_braces_pattern, r'\1+', formatted_rule)
    formatted_rule = re.sub(r'{(\w+)}', r'\1+', formatted_rule)
    
    formatted_rule = tokenize_option(formatted_rule)
    
    # token level
    formatted_tokens = []
    bracket_num = 0
    for token in formatted_rule.split(" "):
        if token == "":
            continue
        processed_token, bracket_num = process_token(token, bracket_num)
        processed_token = post_process_token(processed_token)
        formatted_tokens.append(processed_token)
    grammar = " ".join(formatted_tokens)
    
    grammar = grammar.replace('")")', ')")"')
    if "float : " in grammar:
        grammar = grammar + " | FLOAT_CONSTANT"
    if "int : " in grammar:
        grammar = grammar + " | INT_CONSTANT"
    if "string : " in grammar:
        grammar = grammar + " | STRING"
    if "boolean : " in grammar:
        grammar = grammar + " | BOOLEAN_CONSTANT"
    
    grammar, terminals_dict = post_process_grammar(grammar, terminals_dict)
    lark_grammar.append(grammar)

def parse_grammar(text): 
    lark_grammar = []
    terminals_dict = {}
    formatted_rule = ""
    lines = list(text.splitlines())
    for l_id, line in enumerate(lines):
        if line[:2] == "//" or line == "":
            if formatted_rule != "":
                process_line(formatted_rule, lark_grammar, terminals_dict)
                formatted_rule = ""
        elif "::=" in line:
            if formatted_rule != "":
                process_line(formatted_rule, lark_grammar, terminals_dict)
            formatted_rule = line.strip()
        else:
            formatted_rule += line.strip()
            if line[-1] != "|" and (l_id != len(lines) - 1 and len(lines[l_id+1].strip()) > 0 and lines[l_id+1].strip()[0] != "|"):
                process_line(formatted_rule, lark_grammar, terminals_dict)
                formatted_rule = ""
    
    for key, value in terminals_dict.items():
        lark_grammar.append(f'{key} : {value}')

    lark_grammar.append('seq : "(seq" "{"? moves+ "}"? ")"')
    lark_grammar.append('FLOAT_CONSTANT : /-?\d+(\.\d+)?/ | /-?\.\d+/')
    lark_grammar.append('INT_CONSTANT : /-?\d+/')
    lark_grammar.append('BOOLEAN_CONSTANT : "True" | "False"')
    lark_grammar.append('NUMBER_CONSTANT : FLOAT_CONSTANT | INT_CONSTANT')
    
    for special_nonterminal, rhs_info in SPECIAL_NONTERMINALS.items():
        special_nonterminal_rhs = rhs_info[1]
        lark_grammar.append(f'{special_nonterminal} : {special_nonterminal_rhs}')

    return "\n".join(lark_grammar)

def convert_file_to_lark(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    lark_grammar = parse_grammar(text)
    
    lark_grammar = lark_grammar + "\n\n" + "\n".join(PREFIX)
    
    output_file_path = file_path.replace('.txt', '.lark')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(lark_grammar)
    
    print(f"Converted grammar saved to {output_file_path}")


if __name__ == "__main__":
    convert_file_to_lark('ludii.txt')
