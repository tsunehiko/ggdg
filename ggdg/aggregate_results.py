import json
import argparse
from pathlib import Path

import numpy as np


def aggregate_results(result_dir: Path):
    results = []
    for result_file in result_dir.glob("*/results_dict.json"):
        with result_file.open() as f:
            result = json.load(f)
        results.append(result)
    merged_result = merge_dicts(*results)
    return merged_result


def merge_dicts(*dicts):
    from functools import reduce
    
    def merge_two(d1, d2):
        result = {}
        keys = set(d1.keys()).union(d2.keys())
        for key in sorted(keys):
            if key in d1 and key in d2:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    result[key] = merge_two(d1[key], d2[key])
                else:
                    list1 = d1[key] if isinstance(d1[key], list) else [d1[key]]
                    list2 = d2[key] if isinstance(d2[key], list) else [d2[key]]
                    merged_list = list1 + list2
                    result[key] = merged_list
            elif key in d1:
                if isinstance(d1[key], dict):
                    result[key] = merge_two(d1[key], {})
                else:
                    result[key] = [d1[key]] if not isinstance(d1[key], list) else d1[key]
            else:
                if isinstance(d2[key], dict):
                    result[key] = merge_two({}, d2[key])
                else:
                    result[key] = [d2[key]] if not isinstance(d2[key], list) else d2[key]
        return result

    return reduce(merge_two, dicts, {})


def compute_ids(data):
    if isinstance(data, dict):
        return {key: compute_ids(value) for key, value in data.items()}
    elif isinstance(data, list):
        return sorted(list(set(data)))
    else:
        return data


def compute_stats(data):
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in ["game_ids", "exact_match_id", "demo_copy_game_ids"]:
                result[key] = compute_ids(value)
            else:
                result[key] = compute_stats(value)
        return result
    elif isinstance(data, list):
        n = len(data)
        if n == 0:
            return {'mean': 0.0, 'std': 0.0}
        mean = np.mean(data)
        if n > 1:
            std_dev = np.std(data, ddof=1)
            std_err = std_dev / np.sqrt(n)
        else:
            std_err = 0.0
        return {'mean': mean, 'std': std_err}
    else:
        raise ValueError("Data should be either a dict or a list")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("result_dir", type=str)
    args = argparser.parse_args()
    
    result_dir = Path(args.result_dir)
    results = aggregate_results(result_dir)
    stats_dict = compute_stats(results)
    
    with open(result_dir / "aggregated_results.json", "w") as f:
        json.dump(stats_dict, f, indent=4)
