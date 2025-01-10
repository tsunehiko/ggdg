import re
import multiprocessing
from pathlib import Path

from tqdm import tqdm
from lark import Lark

NUM_MULTIPROCESS = 60

with open("ludii.lark", "r") as file:
    grammar = file.read()
parser = Lark(grammar, start='game')


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

def test_parse(file_path):
    game_name_list = []
    for path_part in list(file_path.parts)[::-1]:
        if path_part == "expand":
            break
        game_name_list.append(path_part)
    game_name = "_".join(game_name_list[::-1])
    game_name = game_name.replace(".txt", "")
    
    with file_path.open('r', encoding='utf-8') as file:
        input_text = file.read()
    
    # input_text_processed = replace_colon_content(input_text)
    input_text_processed = remove_newlines_in_parentheses(input_text)
    
    try:
        tree = parser.parse(input_text_processed)
        print(f"[OK]: {game_name}")
    except Exception as e:
        print(f"[Error]: {game_name}")
        with open(f"parse_errors/{game_name}-output_processed.txt", "w") as file:
            file.write(input_text_processed)
        with open(f"parse_errors/{game_name}-output_origin.txt", "w") as file:
            file.write(input_text)
        with open(f"parse_errors/{game_name}-output_error.txt", "w") as file:
            file.write(str(e))

def main():
    base_path = Path("../../data/ludii/expand")
    file_paths = list(base_path.rglob('*.txt'))
    # for file_path in file_paths:
    #     test_parse(file_path)
    with multiprocessing.Pool(processes=NUM_MULTIPROCESS) as pool:
        pool.map(test_parse, file_paths)


if __name__ == "__main__":
    main()
