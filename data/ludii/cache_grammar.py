import multiprocessing
from pathlib import Path

from minEarley.parser import EarleyParser
from ggdg.utils import * 

SKIP_LIST = ["board/war/replacement/eliminate/target/Mini Wars.lud",
             "board/war/replacement/eliminate/all/Chukaray.lud",
             "board/sow/two_rows/Omangunta Peeta.lud",
             "board/space/connection/Fractal.lud",
             "experimental/Tennessee Waltz.lud"]

NUM_MULTIPROCESS = 60

grammar_file = "grammars/ludii/ludii.lark"
global_parser = EarleyParser.open(grammar_file, start='game', keep_all_tokens=True)

def create_parse_cache(gamefile):
    cache_dir = "data/ludii/grammar"
    save_file_path = cache_dir + "/" + gamefile.replace(".lud", ".txt")
    if Path(save_file_path).exists():
        print(f"[Skipped] {gamefile}")
        return
    
    print(f"[Creating] {gamefile}")
    with open(Path("data/ludii/expand") / gamefile.replace(".lud", ".txt"), "r") as f:
        example_expand_lud = f.read()
    try:
        grammar = lark2bnf(gen_ludii_min_lark(example_expand_lud, global_parser))
    except:
        print(f"[Error] {gamefile}")
        return
    
    Path(save_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_file_path, "w") as f:
        f.write(grammar)

def main():
    gamefiles = []
    with open("data/ludii/gamelist_expand.txt", "r") as f:
        for line in f:
            gamefiles.append(line.strip())
    non_exist_files = []
    
    cache_dir = "data/ludii/grammar"
    for gamefile in gamefiles:
        save_file_path = cache_dir + "/" + gamefile.replace(".lud", ".txt")
        if not Path(save_file_path).exists() and gamefile not in SKIP_LIST:
            non_exist_files.append(gamefile)
    print(f"Number of files to create: {len(non_exist_files)}")
    
    # for gamefile in non_exist_files:
    #     create_parse_cache(gamefile)
    
    with multiprocessing.Pool(processes=NUM_MULTIPROCESS) as pool:
        pool.map(create_parse_cache, non_exist_files)


if __name__ == "__main__":
    main()
