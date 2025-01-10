import json
from pathlib import Path
from functools import partial

import numpy as np
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer

from ggdg.dataset import load_ludii_example
from ggdg.utils import * 


NUM_TRAIN_EXAMPLES = 3 # default: 3
TOKEN_LEN_MAX = 300 # default: 300
TOKEN_LEN_MIN = 0 # default: 0
GAME_DICTS_PATH = "data/ludii/game_dicts/main.json"
TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct" # default: "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_PATH = "data/ludii/analyze/default.json"


def generate_train_examples(test_game, all_examples):
    group = str(Path(test_game).parent)
    other_examples = []
    for example in all_examples:
        if str(Path(example.gamepath).parent) == group:
            continue
        token_length = tokenizer(example.target, return_tensors="pt")["input_ids"].shape[1]
        if token_length > TOKEN_LEN_MAX or token_length < TOKEN_LEN_MIN or not example.target.startswith("(game"):
            continue
        other_examples.append(example)
    if len(other_examples) >= NUM_TRAIN_EXAMPLES:
        train_examples = np.random.choice(other_examples, NUM_TRAIN_EXAMPLES, replace=False)
    else:
        print(f"Game {test_game} has less than {NUM_TRAIN_EXAMPLES} examples")
        train_examples = other_examples
    return test_game, [train_example.gamepath for train_example in train_examples]


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

test_filename_list = "data/ludii/gamelist_grammar.txt"
with open(test_filename_list, "r") as file:
    test_filenames = file.read().splitlines()
all_examples = []
for test_filename in test_filenames:
    example = load_ludii_example(test_filename)
    all_examples.append(example)

with open(GAME_DICTS_PATH, "r") as file:
    game_dicts = json.load(file)
test_games = list(game_dicts.keys())

generate_train_examples_loaded = partial(generate_train_examples, all_examples=all_examples)

train_game_dicts = dict(process_map(generate_train_examples_loaded, test_games, max_workers=10))

with open(SAVE_PATH, "w") as file:
    json.dump(train_game_dicts, file, indent=4)
