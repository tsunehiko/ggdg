import os
import json
import re
import regex
from pathlib import Path
from dataclasses import dataclass
from textwrap import dedent

from minEarley.parser import EarleyParser
from ggdg.flags import FLAGS
from ggdg.utils import * 

@dataclass
class Example:
    source: str
    target: str

    pred_grammar: str = None
    grammar: str = None
    label = None
    gamename: str = None
    gamepath: str = None
    train_examples = None

def load_examples(filename):
    examples = []
    assert len(filename.split(",")) == 2
    src_filename = filename.split(",")[0]
    trg_filename = filename.split(",")[1]
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(Example(source=line1.strip(), target=line2.strip(),))
    return examples

def load_ludii_grammar(filepath: Path):
    with open(filepath, "r") as f:
        grammar = f.read().rstrip('\n')
    return grammar

def extract_key_value_pairs(text):
    """
    Extracts key-value pairs from the given text, where keys are words after "(" and values are the remaining text until the next ")".
    """
    
    pattern = r'\((\w+)\s*(.*?)\)'
    matches = re.findall(pattern, text, re.DOTALL)
    result_dict = {key.strip(): value.strip().strip('"') for key, value in matches}

    graphics = result_dict.get("graphics", "")
    if graphics:
        graphics_dict = extract_key_value_pairs(graphics)
        result_dict["graphics"] = graphics_dict
        
    return result_dict

def load_ludii_meta(ludii_str):
    metadata_match = re.search(r"\(metadata\s*", ludii_str)
    if metadata_match is None:
        return {}
    metadata_str = ludii_str[metadata_match.end():]
    
    metadata_pattern = r"\((?:[^()]|(?R))*\)"
    metadatas = regex.findall(metadata_pattern, metadata_str)
    if len(metadatas) == 0:
        return {}
    info_str = ""
    for metadata in metadatas:
        if metadata[1:5] == "info":
            info_str = metadata
            break
    info_str = info_str[5:-1].strip()
    
    info_pattern = r"\((?:[^()]|(?R))*\)"
    infos = regex.findall(info_pattern, info_str)
    metadata_dict = {}
    for info in infos:
        data = info.split('"')
        if len(data) < 2:
            continue
        
        key = data[0][1:].strip()
        value = '"'.join(data[1:-1])
        value = re.sub(r"\s+", " ", value)
        if key in metadata_dict:
            metadata_dict[key] += "\n" + value
        else:
            metadata_dict[key] = value
        
        if "useFor_rules" not in metadata_dict and key == "useFor" \
            and ("Reconstructed" in value or "Suggested" in value or "Described" in value or "Observed" in value or "Historical Information" in value):
            use_for_pattern = r"\((?:[^()]|(?R))*\)"
            use_for_list = regex.findall(use_for_pattern, value + ")")
            for use_for in use_for_list:
                if "rules" in use_for:
                    metadata_dict["useFor_rules"] = use_for[6:-1].strip()
                    break
    
    return metadata_dict

def load_ludii_from_file(filepath: Path):
    ludii, metadata = "", ""
    metadata_start = False
    with open(filepath, "r") as f:
        for line in f:
            if not metadata_start and line.startswith("//"):
                metadata_start = True
            elif metadata_start:
                metadata += line
            else:
                ludii += line
    metadata_dict = load_ludii_meta(metadata)
    return ludii, metadata_dict

def replace_colon_content(file_content):
    updated_content = file_content
    bracket_pattern = re.compile(r'\b\w+:\(([^)]+)\)')
    non_bracket_pattern = re.compile(r'\b\w+:([^\s]+)')

    while True:
        if not bracket_pattern.findall(updated_content) and not non_bracket_pattern.findall(updated_content):
            break
        updated_content = bracket_pattern.sub(r'(\1)', updated_content)
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

def load_ludii_examples(filename, num_shot=-1):
    examples = []
    with open(filename, "r") as f:
        test_info = json.load(f)
    for gamepath, trainpaths in test_info.items():
        test_example = load_ludii_example(gamepath)
        train_examples = []
        if num_shot != -1:
            trainpaths = trainpaths[:num_shot]
        for trainpath in trainpaths:
            train_example = load_ludii_example(trainpath)
            train_examples.append(train_example)
        test_example.train_examples = train_examples
        examples.append(test_example)
    return examples

def load_ludii_example(gamepath):
    gamepath = gamepath.strip()
    gamename = gamepath.strip().split(".")[0].replace("/", "_")
    lud, metadata = load_ludii_from_file(Path("data/ludii/lud") / gamepath)
    with open(Path("data/ludii/expand") / gamepath.replace(".lud", ".txt"), "r") as f:
        expand_lud = f.read()
    expand_lud = remove_newlines_in_parentheses(expand_lud)
    with open(Path("data/ludii/grammar") / gamepath.replace(".lud", ".txt"), "r") as f:
        grammar = f.read()
    grammar = grammar.strip()
    source = {}
    if "description" in metadata:
        source["description"] = dedent(metadata["description"])
    if "rules" in metadata:
        source["rules"] = dedent(metadata["rules"])
    elif "useFor_rules" in metadata:
        source["rules"] = dedent(metadata["useFor_rules"])
    target = expand_lud.strip()
    example = Example(source=source, target=target, gamename=gamename, grammar=grammar, gamepath=gamepath)
    return example
    

def load_sempar_data(config):
    if config["dataset"] == "ludii":
        filename_list = config["game_list_path"]
        examples = load_ludii_examples(filename_list, num_shot=config["num_shot"])

        test_num = config["test_num"]
        if test_num != -1:
            examples = examples[:test_num]

    else:
        raise ValueError(f"dataset {config['dataset']} not supported")
    return examples

def load_sem_parser(config):
    if config["dataset"] == "ludii":
        grammar_file = "grammars/ludii/ludii.lark"
        global_parser = EarleyParser.open(grammar_file, start='game', keep_all_tokens=True)
    else:
        raise ValueError(f"dataset {config['dataset']} not supported")
    raw_global_rules, _ = collect_rules_from_larkfile(grammar_file)
    global_rules = collect_rules_from_parser(global_parser)
    return global_parser, global_rules, raw_global_rules

def counter2pred(counter):
    if len(counter) == 0:
        return None
    else:
        return counter.most_common(1)[0][0]
