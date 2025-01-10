SEED=0

python ggdg/main.py \
    --seed $SEED \
    --dataset ludii \
    --engine hf/meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --output_max_tokens 2048 \
    --input_max_tokens 8192 \
    --prompt_mode grammar \
    --game_list_path "data/ludii/game_dicts/main.json" \
    --test_num 100 \
    --exp_name main/random/${SEED} \
    --random_sampling \
