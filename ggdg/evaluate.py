import argparse
import json
import subprocess
import re
from pathlib import Path
from collections import defaultdict
from functools import partial

import torch.multiprocessing as multiprocessing
if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from nltk.translate import meteor_score
from nltk import word_tokenize

from bert_score import score as bscore
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ggdg.utils import *
from ggdg.dataset import load_ludii_example
from ggdg.train_utils import logger, setup_logger_file


CONCEPTS_GT_DIR = "data/ludii/concepts"


def counter2pred(counter):
    if len(counter) == 0:
        return None
    else:
        return counter.most_common(1)[0][0]


def evaluate_programs(predictions_dir: Path):
    if len(list(predictions_dir.iterdir())) == 0:
        return 0.0

    counter = 0
    for prediction_file in predictions_dir.iterdir():
        with open(prediction_file, "r") as f:
            prediction = f.read().strip()
        result_info_path = str(prediction_file.parent.parent / "info" / prediction_file.name).replace(".lud", ".json")
        with open(result_info_path, "r") as f:
            info = json.load(f)
        gamepath = info["gamepath"]
        example = load_ludii_example(gamepath)
        target = example.target.strip()
        prediction = re.sub(r'\s{2,}', ' ', prediction.replace("\n", ""))
        target = re.sub(r'\s{2,}', ' ', target.replace("\n", ""))
        if prediction == target:
            counter += 1
    return counter / len(list(predictions_dir.iterdir()))


def evaluate_llm_call(info_dir: Path):
    llm_call_count_rule_list, llm_call_count_program_list, llm_call_count_list = [], [], []
    for info_file in info_dir.iterdir():
        with open(info_file, "r") as f:
            info = json.load(f)
        llm_call_dict = info["llm_call_count"]
        llm_call_count_rule_list.append(llm_call_dict["rule"])
        llm_call_count_program_list.append(llm_call_dict["program"])
        llm_call_count_list.append(llm_call_dict["total"])
    return {
        "llm_call_count": {
            "rule": llm_call_count_rule_list,
            "program": llm_call_count_program_list,
            "total": llm_call_count_list
        }
    }


def evaluate_ludii_grammars(prediction_dir: Path):
    if len(list(prediction_dir.iterdir())) == 0:
        return {
            "grammar": 
                {
                    "precision": [0.0],
                    "recall": [0.0],
                    "f_one": [0.0]
                }
            }
    
    precisions, recalls, f_ones = [], [], []
    for prediction_file in tqdm(list(prediction_dir.iterdir()), desc="Checking Ludii grammars", leave=False):
        with open(prediction_file, "r") as f:
            prediction = f.read()
        
        if len(prediction) == 0 or prediction == "None":
            continue
        
        else:
            result_info_path = str(prediction_file.parent.parent / "info" / prediction_file.name).replace(".txt", ".json")
            with open(result_info_path, "r") as f:
                info = json.load(f)
            gamepath = info["gamepath"]
            example = load_ludii_example(gamepath)
            gt_lark_grammar = example.grammar
            # gt_lhs_raw_rhs_dict = get_lhs_raw_rhs_dict(gt_lark_grammar)
            gt_lark_grammar = bnf2lark(gt_lark_grammar)
            gt_lark_grammar = ebnflark2bnflark(gt_lark_grammar)
            # local_parser = EarleyParser(gt_lark_grammar, start="game")
            # gt_rules_set = collect_rules_from_parser(local_parser)
            gt_rules_set = list(set([rule for rule in larkstr2rulelist(gt_lark_grammar)]))
            
            # pred_lhs_raw_rhs_dict = get_lhs_raw_rhs_dict(prediction)
            pred_lark_grammar = bnf2lark(prediction)
            pred_lark_grammar = ebnflark2bnflark(pred_lark_grammar)
            correct_count = 0
            pred_rules, valid_rules = [], []
            # pred_lhs_rhs_dict = collections.defaultdict(list)
            for raw_rule_str in pred_lark_grammar.split("\n"):
                if ":" not in raw_rule_str:
                    continue
                lhs, rhs_str = split_rule(raw_rule_str)
                rhs_options = get_rhs_options(rhs_str)
                for rhs_tokens in rhs_options:
                    raw_rule = SimpleRule(lhs, tuple(rhs_tokens))
                    short_rhs_tokens = get_short_tokens(rhs_tokens)
                    short_rule = SimpleRule(lhs, tuple(short_rhs_tokens))
                    pred_rules.append(short_rule)
                    if short_rule in gt_rules_set or raw_rule in gt_rules_set:
                        valid_rules.append(short_rule)
            # for rule in larkstr2rulelist(pred_lark_grammar):
            #     if rule in pred_rules:
            #         continue
            #     pred_rules.append(rule)
            #     lhs = rule.origin
            #     if lhs not in gt_lhs_raw_rhs_dict:
            #         continue
            #     if pred_lhs_raw_rhs_dict[lhs] == gt_lhs_raw_rhs_dict[lhs] or rule in gt_rules_set:
            #         correct_count += 1
            pred_rules_set = list(set(pred_rules))
            correct_count = len(list(set(valid_rules)))
            
            precision = correct_count / len(pred_rules_set) if len(pred_rules_set) > 0 else 0
            recall = correct_count / len(gt_rules_set) if len(gt_rules_set) > 0 else 0
        f_one = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f_ones.append(f_one)
    
    return {
        "grammar": {
            "precision": precisions,
            "recall": recalls,
            "f_one": f_ones
        }
    }


def extract_section(game, section):
    pattern = re.compile(r'\(' + section + r'\s')
    match = pattern.search(game)
    if not match:
        return ""
    
    start = match.end()
    open_braces = 1
    end = start
    
    while open_braces > 0 and end < len(game):
        if game[end] == '(':
            open_braces += 1
        elif game[end] == ')':
            open_braces -= 1
        end += 1
    
    return game[start:end-1].strip()


# BLEU@4
def get_blue_scores(prediction, target):
    candidate = prediction.split()
    reference = [target.split()]
    chencherry = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=chencherry) * 100
    return bleu_score


# ROUGE-1, ROUGE-2, ROUGE-L
def get_rouge_scores(prediction, target):
    rouge = Rouge()
    if prediction == "":
        return 0.0, 0.0, 0.0
    else:
        try:
            rouges = rouge.get_scores(prediction, target)
        except:
            return 0.0, 0.0, 0.0
        rouge_l_1 = np.mean([score["rouge-1"]["f"] for score in rouges]) * 100
        rouge_l_2 = np.mean([score["rouge-2"]["f"] for score in rouges]) * 100
        rouge_l_f1 = np.mean([score["rouge-l"]["f"] for score in rouges]) * 100
        return rouge_l_1, rouge_l_2, rouge_l_f1


# METEOR
def get_meteor_scores(prediction, target):
    if prediction == "":
        return 0.0
    else:
        reference_tokens = word_tokenize(target)
        prediction_tokens = word_tokenize(prediction)
        meteor = meteor_score.meteor_score([reference_tokens], prediction_tokens) * 100
        return meteor


# BERTScore
def get_bert_score(prediction, target):
    if prediction == "":
        return 0.0, 0.0, 0.0
    P, R, F1 = bscore([prediction], [target], lang="en", verbose=False)
    return P[0].item() * 100, R[0].item() * 100, F1[0].item() * 100


# Exact match
def get_exact_match(prediction, target):
    return float(prediction == target)


def get_nlp_scores(prediction, target, bert_score=False):
    bleu_score = get_blue_scores(prediction, target)
    rouge_l_1, rouge_l_2, rouge_l_f1 = get_rouge_scores(prediction, target)
    meteor_score = get_meteor_scores(prediction, target)
    exact_match = get_exact_match(prediction, target)
    levenshtein_distance = Levenshtein.distance(prediction, target)
    nlp_score_results = {
        "bleu_score": bleu_score,
        "rouge_l_1": rouge_l_1,
        "rouge_l_2": rouge_l_2,
        "rouge_l_f1": rouge_l_f1,
        "meteor_score": meteor_score,
        "exact_match": exact_match,
        "levenshtein_distance": levenshtein_distance
    }
    if bert_score:
        bert_score = get_bert_score(prediction, target)
        nlp_score_results["bert_score"] = bert_score
    return nlp_score_results


def process_nlp_scores(input, bert_score=False):
    game_id, prediction, target = input
    return {
        "game_id": game_id,
        "Full": get_nlp_scores(prediction, target, bert_score),
        "Players": get_nlp_scores(extract_section(prediction, "players"), extract_section(target, "players"), bert_score),
        "Equipment": get_nlp_scores(extract_section(prediction, "equipment"), extract_section(target, "equipment"), bert_score),
        "Rules": get_nlp_scores(extract_section(prediction, "rules"), extract_section(target, "rules"), bert_score)
    }


def evaluate_nlp_scores(predictions_dir: Path, num_processes: int, bert_score: bool = False, demo_copy_game_ids: list = []):
    all_results = {
        "Full": defaultdict(list),
        "Players": defaultdict(list),
        "Equipment": defaultdict(list),
        "Rules": defaultdict(list)
    }
    nlp_inputs = []
    for prediction_file in predictions_dir.iterdir():
        game_id = int(prediction_file.stem.split("_")[0])
        if game_id in demo_copy_game_ids:
            continue
        with open(prediction_file, "r") as f:
            prediction = f.read()
        result_info_path = str(prediction_file.parent.parent / "info" / prediction_file.name).replace(".lud", ".json")
        if not Path(result_info_path).exists():
            continue
        with open(result_info_path, "r") as f:
            info = json.load(f)
        gamepath = info["gamepath"]
        example = load_ludii_example(gamepath)
        target = example.target
        prediction = re.sub(r'\s{2,}', ' ', prediction.replace("\n", ""))
        target = re.sub(r'\s{2,}', ' ', target.replace("\n", ""))
        nlp_inputs.append((game_id, prediction, target))
    
    partial_process_nlp_scores = partial(process_nlp_scores, bert_score=bert_score)
    results = process_map(partial_process_nlp_scores, nlp_inputs, max_workers=num_processes)
    
    other_info = {"rouge_raw_score": {}, "exact_match_id": []}
    for result in results:
        result_game_id = result["game_id"]
        for section, scores in result.items():
            if section == "game_id":
                continue
            for key, value in scores.items():
                all_results[section][key].append(value)
                if section == "Full" and key == "rouge_l_f1":
                    other_info["rouge_raw_score"][result_game_id] = value
                if section == "Full" and key == "exact_match" and value == 1:
                    other_info["exact_match_id"].append(result_game_id)
    
    return {"nlp_scores": all_results}, other_info


def log_transform(x):
    return np.log2(x + 1) if x >= 0 else -np.log2(1 - x)


# DROP_CONCEPTS = [
#     "Efficiency", "CopyContext", "Then", "ForEachPiece", "DoLudeme", "Trigger", "PlayoutsPerSecond", "MovesPerSecond"
# ]

def get_board_distance(board1_csv, board2_csv):
    df1 = pd.read_csv(board1_csv, nrows=1)
    # df1 = df1.drop(columns=DROP_CONCEPTS, errors='ignore')
    data1 = df1.iloc[0]
    df2 = pd.read_csv(board2_csv, nrows=1)
    # df2 = df2.drop(columns=DROP_CONCEPTS, errors='ignore')
    data2 = df2.iloc[0]
    mask = (data1.notna() & data2.notna() & 
            data1.apply(lambda x: pd.api.types.is_numeric_dtype(type(x))) & 
            data2.apply(lambda x: pd.api.types.is_numeric_dtype(type(x))))
    vector1 = data1[mask].astype(float).values
    vector1 = np.vectorize(log_transform)(vector1)
    vector2 = data2[mask].astype(float).values
    vector2 = np.vectorize(log_transform)(vector2)
    cosine_dist = cosine(vector1, vector2) / 2
    euclidean_dist = euclidean(vector1, vector2) / np.sqrt(vector1.shape[0])
    return cosine_dist, euclidean_dist


def evaluate_ludii_game(evaluate_args, timeout=1800):
    prediction_file, intermediate_dir, mode = evaluate_args

    game_intermediate_dir = intermediate_dir / prediction_file.stem
    result_json_path = game_intermediate_dir / "base_result.json"
    trial_dir = game_intermediate_dir / "trial"
    if mode == "full":
        concept_dir = game_intermediate_dir / "concept"
    elif mode == "short":
        concept_dir = game_intermediate_dir / "concept_short"
    elif mode == "none":
        concept_dir = game_intermediate_dir / "concept_none"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    concept_csv = concept_dir / "Concepts.csv"
    game_id = int(prediction_file.stem.split("_")[0])
    ludii_game_result_dict = {"game_id": game_id}
    
    with open(prediction_file, "r") as f:
        prediction = f.read()
    if len(prediction) == 0:
        return ludii_game_result_dict
    prediction_info_path = str(prediction_file.parent.parent / "info" / prediction_file.name).replace(".lud", ".json")
    if not Path(prediction_info_path).exists():
        return ludii_game_result_dict
    with open(prediction_info_path, "r") as f:
        info = json.load(f)
    gamepath = info["gamepath"]
    
    player_count_str_list = re.findall(r'\(players (\d+)\)', prediction)
    player_count = int(player_count_str_list[0]) if len(player_count_str_list) > 0 else 0
    
    if not result_json_path.exists():
        game_intermediate_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(['java', '-jar', 'ludii_java/jars/EvalLudiiGame.jar', '--game-path', str(prediction_file),
                                     ], capture_output=True, text=True, timeout=timeout)
            result_str_list = re.findall(r"\{'isCompilable':.*?\}", result.stdout)
            if len(result_str_list) == 0:
                result_dict = {"isCompilable": "false", "isFunctional": "false", "isPlayable": "false"}
            else:
                result_dict = json.loads(result_str_list[0].strip().replace("'", '"'))
        except Exception as e:
            logger.error(f"failed to check ludii game: {prediction_file}")
            logger.error(str(e))
            result_dict = {"isCompilable": "false", "isFunctional": "false", "isPlayable": "false"}
        with open(result_json_path, "w") as f:
            json.dump(result_dict, f)
    
    else:
        with open(result_json_path, "r") as f:
            result_dict = json.load(f)
    
    if "isCompilable" not in result_dict or "isFunctional" not in result_dict or "isPlayable" not in result_dict:
        logger.error(f"failed to parse ludii game result: {prediction_file}")
        return result_dict
    
    if result_dict["isCompilable"] == "true":
        ludii_game_result_dict["ludii_compilable"] = 1
    if result_dict["isFunctional"] == "true":
        ludii_game_result_dict["functional"] = 1
        
        if mode != "none":
            if not concept_csv.exists():
                try:
                    concept_short_str = "true" if mode == "short" else "false"
                    result = subprocess.run(['java', '-jar', 'ludii_java/jars/ComputeConcept.jar', '--trials-dir', trial_dir,
                                            '--concepts-dir', concept_dir, '--game-path', str(prediction_file),
                                            '--num-threads', '10', '--num-trials', '10', '--short', concept_short_str,
                                            ], capture_output=True, text=True, timeout=timeout)
                except Exception as e:
                    logger.error(f"failed to check ludii game: {prediction_file}")
                    logger.error(str(e))
            
            if concept_csv.exists():
                # Extract game information from Concepts.csv
                df = pd.read_csv(concept_csv)
                agency = get_column_value(df, "DecisionMoves")
                if agency is not None:
                    ludii_game_result_dict["agency"] = agency
                coverage = get_column_value(df, "BoardCoverageUsed")
                if coverage is not None:
                    ludii_game_result_dict["coverage"] = coverage
                timeouts = get_column_value(df, "Timeouts")
                if timeouts is not None:
                    completion = 1. - timeouts
                    ludii_game_result_dict["completion"] = completion
                if player_count > 1:
                    balance = get_column_value(df, "Balance")
                    if balance is not None:
                        ludii_game_result_dict["balance"] = balance
                    decisiveness = get_column_value(df, "Completion")
                    if decisiveness is not None:
                        ludii_game_result_dict["decisiveness"] = decisiveness
                
                # Board distance
                gt_concepts_csv = Path(CONCEPTS_GT_DIR) / gamepath.split(".")[0] / "Concepts.csv"
                if mode == "full" and gt_concepts_csv.exists():
                    cosine_distance, euclidean_distance = get_board_distance(concept_csv, gt_concepts_csv)
                    with open(concept_dir / "distance.json", "w") as f:
                        json.dump({"cosine": cosine_distance, "euclidean": euclidean_distance}, f)
                    ludii_game_result_dict["cosine_distance"] = cosine_distance
                    ludii_game_result_dict["euclidean_distance"] = euclidean_distance
                
                # Multi-agent evaluation
                if player_count > 1:
                    multiagent_eval_dir = game_intermediate_dir / "multiagent_eval"
                    multiagent_result_path = multiagent_eval_dir / "results.txt"
                    if not multiagent_result_path.exists():
                        multiagent_eval_dir.mkdir(parents=True, exist_ok=True)
                        with open("ludii_java/src/multiagent_evalgame_config.json", "r") as f:
                            config = json.load(f)
                        config["gameName"] = str(prediction_file)
                        config["outDir"] = str(multiagent_eval_dir)
                        config["agentStrings"] = ["MCTS"] + ["Random" for _ in range(player_count - 1)]
                        with open(multiagent_eval_dir / "config.json", "w") as f:
                            json.dump(config, f)
                        try:
                            subprocess.run(['java', '-jar', './ludii_java/jars/ComputeMultiAgents.jar',
                                            '--json-files', str(multiagent_eval_dir / "config.json")])
                        except Exception as e:
                            logger.error(f"failed to evaluate multiagent ludii game: {prediction_file}")
                            logger.error(str(e))

                    if multiagent_result_path.exists():
                        with open(multiagent_eval_dir / "results.txt", "r") as f:
                            multiagent_eval_results = f.read()
                        mcts_win_rate_str_list = re.findall(r"Agent 1 \(UCT\)[\s\S]*?mean=([\d.]+)", multiagent_eval_results)
                        mcts_win_rate = float(mcts_win_rate_str_list[0]) if len(mcts_win_rate_str_list) > 0 else 0
                        ludii_game_result_dict["mcts_win_rate"] = mcts_win_rate
    
    elif mode != "none":
        ludii_game_result_dict["cosine_distance"] = 1.0
    
    if result_dict["isPlayable"] == "true":
        ludii_game_result_dict["playable"] = 1
    
    return ludii_game_result_dict


def multiprocess_evaluate_ludii_games(prediction_dir: Path, mode: str = "full", num_processes: int = 10, 
                                      prediction_num: int = 0, timeout: int = 1800):
    intermediate_dir = prediction_dir.parent.parent / "eval" / "ludii_eval"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    input_args = [(prediction_file, intermediate_dir, mode) for prediction_file in prediction_dir.iterdir()]
    partial_evaluate_ludii_game = partial(evaluate_ludii_game, timeout=timeout)
    result_list = process_map(partial_evaluate_ludii_game, input_args, max_workers=num_processes)
    # result_list = []
    # for prediction_file in tqdm(list(prediction_dir.iterdir()), desc="Checking Ludii games", leave=False):
    #     result = evaluate_ludii_game((prediction_file, intermediate_dir, mode))
    #     result_list.append(result)
    results, game_ids = defaultdict(list), defaultdict(list)
    for result in result_list:
        game_id = result["game_id"]
        for key, value in result.items():
            if key == "game_id":
                continue
            if key in ["ludii_compilable", "functional", "playable"]:
                game_ids[key].append(game_id)
            else:
                results[key].append(value)
    for key, value in game_ids.items():
        results[key] = len(value)
        results["rate_" + key] = len(value) * 100 / prediction_num if prediction_num > 0 else 0
        game_ids[key] = sorted(value)
    if "ludii_compilable" not in results:
        results["ludii_compilable"] = 0
        results["rate_ludii_compilable"] = 0
    if "functional" not in results:
        results["functional"] = 0
        results["rate_functional"] = 0
    if "playable" not in results:
        results["playable"] = 0
        results["rate_playable"] = 0
    return dict(results), dict(game_ids)


def get_demo_copy_and_distance(prediction: str, demo_programs: list):
    if len(demo_programs) == 0:
        return 0, [0.]
    gamename_pattern = r'"([^"]*)"'
    pred_gamename_match = re.search(gamename_pattern, prediction)
    if pred_gamename_match is None:
        pred_gamename = ""
    else:
        pred_gamename = pred_gamename_match.group(1)
    prediction = prediction.replace(pred_gamename, "").strip()
    prediction = re.sub(r'\s{2,}', ' ', prediction.replace("\n", ""))
    
    exact_match, distances = [], []
    for demo_program in demo_programs:
        demo_program = demo_program.strip()
        demo_gamename_match = re.search(gamename_pattern, demo_program)
        if demo_gamename_match is None:
            continue
        else:
            demo_gamename = demo_gamename_match.group(1)
        demo_program = demo_program.replace(demo_gamename, "").strip()
        demo_program = re.sub(r'\s{2,}', ' ', demo_program.replace("\n", ""))
        if prediction == demo_program:
            exact_match.append(1)
        else:
            exact_match.append(0)
        distances.append(Levenshtein.distance(prediction, demo_program))
    
    return max(exact_match), distances


def evaluate_demonstration(prediction_dir: Path, prediction_num: int):
    demo_copy_game_ids, pred_demo_min_distances = [], []
    gt_demo_distances = []
    for prediction_file in prediction_dir.iterdir():
        game_id = int(prediction_file.stem.split("_")[0])
        
        # Demo copy
        with open(prediction_file, "r") as f:
            prediction = f.read()
        demo_file_path = prediction_file.parent.parent.parent / "demo" / str(prediction_file.name).replace(".lud", ".txt")
        if not demo_file_path.exists():
            continue
        with open(demo_file_path, "r") as f:
            demonstration = f.read()
        program_pattern = r'<program>(.*?)</program>'
        demo_programs = re.findall(program_pattern, demonstration, re.DOTALL)
        demo_exact_match, pred_demo_distances = get_demo_copy_and_distance(prediction, demo_programs)
        if demo_exact_match > 0:
            demo_copy_game_ids.append(game_id)
        pred_demo_min_distances.append(min(pred_demo_distances))

        # GT demo distance
        gt_file_path = prediction_file.parent.parent.parent / "gt" / "program" / str(prediction_file.name).replace(".lud", ".txt")
        with open(gt_file_path, "r") as f:
            target_program = f.read()
        _, pred_gt_distances = get_demo_copy_and_distance(target_program, demo_programs)
        gt_demo_distances.append(np.mean(pred_gt_distances))
    return {
        "demonstration": {
            "demo_copy_games": len(demo_copy_game_ids),
            "rate_demo_copy_games": len(demo_copy_game_ids) * 100 / prediction_num if prediction_num > 0 else 0,
            "pred_demo_min_distance": np.mean(pred_demo_min_distances),
            "gt_demo_distance": np.mean(gt_demo_distances)
        }
    }, demo_copy_game_ids


def get_result_value(value):
    if type(value) == list:
        if len(value) == 0:
            return {"mean": 0.0, "std": 0.0}
        mean = np.mean(value)
        std = np.std(value) / np.sqrt(len(value))
        return {"mean": mean, "std": std}
    elif type(value) == dict or type(value) == defaultdict:
        return {key: get_result_value(val) for key, val in value.items()}
    else:
        return value


def print_result(key, value, indent=0):
    if type(value) == dict or type(value) == defaultdict:
        print_str = "    " * indent + f"{key}:"
        logger.info(print_str)
        for k, v in value.items():
            print_result(k, v, indent + 1)
    else:
        print_str = "    " * indent + f"{key}: {value:.4f}"
        logger.info(print_str)


def add_result_to_dict(raw_dict, results_dict):
    for key, value in raw_dict.items():
        results_dict[key] = get_result_value(value)
        print_result(key, results_dict[key])
    return results_dict


def evaluate(log_dir: Path, mode: str = "full", num_processes: int = 10, 
             bert_score: bool = False, timeout: int = 1800):
    results_dict = {}

    # llm call count evaluation
    logger.info("-"*10 + " Evaluating LLM call count " + "-"*10)
    prediction_info_dir = log_dir / "prediction" / "info"
    llm_call_results = evaluate_llm_call(prediction_info_dir)
    results_dict = add_result_to_dict(llm_call_results, results_dict)

    # grammar evaluation
    logger.info("-"*10 + " Evaluating Grammar " + "-"*10)
    prediction_grammar_dir = log_dir / "prediction" / "grammar"
    if prediction_grammar_dir.exists():
        grammar_results = evaluate_ludii_grammars(prediction_grammar_dir)
        results_dict = add_result_to_dict(grammar_results, results_dict)
    
    prediction_program_dir = log_dir / "prediction" / "program"
    prediction_num = len(list(prediction_program_dir.iterdir()))
    results_dict["prediction_num"] = prediction_num
    
    # demonstration evaluation
    logger.info("-"*10 + " Evaluating Demonstration " + "-"*10)
    demo_results, demo_copy_game_ids = evaluate_demonstration(prediction_program_dir, prediction_num)
    results_dict = add_result_to_dict(demo_results, results_dict)

    # NLP evaluation
    logger.info("-"*10 + " Evaluating NLP " + "-"*10)
    nlp_results, nlp_other_info = evaluate_nlp_scores(prediction_program_dir, num_processes, bert_score)
    results_dict = add_result_to_dict(nlp_results, results_dict)
    rouge_raw_score = {k: v for k, v in sorted(nlp_other_info["rouge_raw_score"].items(), key=lambda item: item[1], reverse=True)}
    results_dict["rouge_raw_score"] = rouge_raw_score

    # ludii evaluation
    logger.info("-"*10 + " Evaluating Ludii " + "-"*10)
    logger.info(f"Mode: {mode}")
    ludii_results, game_ids = multiprocess_evaluate_ludii_games(prediction_program_dir, mode, num_processes, 
                                                                prediction_num, timeout=timeout)
    results_dict = add_result_to_dict(ludii_results, results_dict)
    
    # game ids
    game_ids["demo_exact_match"] = sorted(demo_copy_game_ids)
    game_ids["exact_match_id"] = nlp_other_info["exact_match_id"]
    results_dict["game_ids"] = game_ids

    with open(log_dir / "results_dict.json", "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("log_dir", type=str)
    argparser.add_argument("--mode", type=str, default="full")
    argparser.add_argument("--bert-score", action="store_true")
    argparser.add_argument("--num-processes", type=int, default=10)
    argparser.add_argument("--timeout", type=int, default=1800)
    args = argparser.parse_args()

    log_dir = Path(args.log_dir)
    logger_dir = log_dir / "eval"
    setup_logger_file(logger, logger_dir)
    
    evaluate(log_dir, args.mode, args.num_processes, args.bert_score, args.timeout)
