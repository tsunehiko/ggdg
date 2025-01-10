import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from ggdg.dataset import load_ludii_example
from ggdg.utils import * 


def trim_outliers(data, lower_percentile=5, upper_percentile=95):
    data = np.array(data)
    
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    trimmed_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return trimmed_data


def plot_histogram(data, title, filename):
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.75, color='blue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('length')
    ax.set_ylabel('Frequency')
    plt.savefig(filename)
    plt.close(fig)


NUM_GAMES = 100 # default: 100
NUM_TRAIN_EXAMPLES = 3 # default: 3
TOKEN_LEN_MAX = 300 # default: 300
TOKEN_LEN_MIN = 0 # default: 0
TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct" # default: "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_PATH = "data/ludii/analyze/default.json"
GROUP = "" # default: "", example: "board/sow/two_rows"
IS_DIFFERENT_CATEGORY = False # default: False


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

test_filename_list = "data/ludii/gamelist_grammar.txt"
with open(test_filename_list, "r") as file:
    test_filenames = file.read().splitlines()
test_examples = []
for test_filename in test_filenames:
    example = load_ludii_example(test_filename)
    test_examples.append(example)

target_length_list = []
target_length_by_group = defaultdict(list)
for test_example in tqdm(test_examples):
    target_token_length = tokenizer(test_example.target, return_tensors="pt")["input_ids"].shape[1]
    if target_token_length > TOKEN_LEN_MAX or target_token_length < TOKEN_LEN_MIN \
        or not test_example.target.startswith("(game"):
        continue
    group = str(Path(test_example.gamepath).parent)
    if GROUP != "" and not group.startswith(GROUP):
        continue
    target_length_by_group[group].append(test_example)

use_game_dict = {}
if IS_DIFFERENT_CATEGORY:
    all_groups = list(target_length_by_group.keys())
    for group_name, test_example_group in tqdm(list(target_length_by_group.items())):
        other_group_games = []
        for other_group_name in all_groups:
            if other_group_name != group_name:
                other_group_games.extend(target_length_by_group[other_group_name])
        for e_id, test_example in enumerate(test_example_group):
            train_examples = np.random.choice(other_group_games, NUM_TRAIN_EXAMPLES, replace=False)
            use_game_dict[test_example.gamepath] = [train_example.gamepath for train_example in train_examples]
else:
    for _, test_example_group in tqdm(list(target_length_by_group.items())):
        if len(test_example_group) < NUM_TRAIN_EXAMPLES + 1:
            continue
        for e_id, test_example in enumerate(test_example_group):
            others = np.delete(test_example_group, e_id)
            train_examples = np.random.choice(others, NUM_TRAIN_EXAMPLES, replace=False)
            use_game_dict[test_example.gamepath] = [train_example.gamepath for train_example in train_examples]

use_game_test_examples = list(use_game_dict.keys())
if len(use_game_test_examples) < NUM_GAMES:
    random_use_game_dict = use_game_dict
else:
    random_keys = random.sample(use_game_test_examples, NUM_GAMES)
    random_use_game_dict = {key: use_game_dict[key] for key in random_keys}

print(f"Number of games: {len(random_use_game_dict)}")
with open(SAVE_PATH, "w") as file:
    json.dump(random_use_game_dict, file, indent=4)
