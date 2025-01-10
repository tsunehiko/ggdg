import json
import subprocess
import multiprocessing
from pathlib import Path

from tqdm import tqdm

from ggdg.dataset import load_ludii_example


GAME_DICT_DIR = "data/ludii/game_dicts"
TEST_FILENAME_LIST = "data/ludii/gamelist_grammar.txt"
CONCEPTS_DIR = "data/ludii/concepts"


def extract_concept(test_filename):
    print(f"{test_filename}")
    
    example = load_ludii_example(test_filename)
    game_fullname = example.gamepath.split(".")[0]
    game_fullpath = Path("data/ludii/lud") / example.gamepath
    trial_dir = Path("data/ludii/trials") / game_fullname
    concept_dir = Path("data/ludii/concepts") / game_fullname
    
    result = subprocess.run(['java', '-jar', 'ludii_java/EvalLudiiGame.jar', '--trials-dir', str(trial_dir),
                             '--concepts-dir', str(concept_dir), '--game-path', str(game_fullpath),
                             '--num-threads', '50', '--num-trials', '100'], capture_output=True, text=True)
    return result


def get_all_json_files(directory):
    path = Path(directory)
    json_files = list(path.rglob('*.json'))
    json_files = [str(file.absolute()) for file in json_files]
    return json_files


if __name__ == "__main__":
    # num_processes = 15
    
    with open(TEST_FILENAME_LIST, "r") as file:
        test_filenames = file.read().splitlines()
    
    concepts_filenames = [test_filename for test_filename in test_filenames 
                          if not (Path(CONCEPTS_DIR) / test_filename.replace(".lud", "") / "Concepts.csv").exists()]
    
    jsonfiles = get_all_json_files(GAME_DICT_DIR)
    use_games = []
    for jsonfile in jsonfiles:
        with open(jsonfile, "r") as file:
            game_dict = json.load(file)
        use_games += list(game_dict.keys())
    use_games = list(set(use_games))
    
    use_games_concepts_filenames = list(set(concepts_filenames) & set(use_games))
    unused_games_concepts_filenames = list(set(concepts_filenames) - set(use_games))
    
    # print(f"Use games: {len(use_games_concepts_filenames)}")
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(extract_concept, use_games_concepts_filenames)
    # print("Use games are done!")
    
    # print(f"Unused games: {len(unused_games_concepts_filenames)}")
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(extract_concept, unused_games_concepts_filenames)
    # print("Unused games are done!")
    
    print(f"Use games: {len(use_games_concepts_filenames)}")
    for concept_filename in tqdm(use_games_concepts_filenames):
        result = extract_concept(concept_filename)
    print("Use games are done!")
    
    print(f"Unused games: {len(unused_games_concepts_filenames)}")
    for concept_filename in tqdm(unused_games_concepts_filenames):
        result = extract_concept(concept_filename)
    print("Unused games are done!")
