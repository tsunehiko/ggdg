SEED=0

python ggdg/main.py \
    --seed $SEED \
    --exp_name main/gdg/${SEED} \
    --dataset ludii \
    --engine hf/meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --output_max_tokens 2048 \
    --input_max_tokens 8192 \
    --prompt_mode std \
    --game_list_path "data/ludii/game_dicts/main.json" \
    --test_num 100 \
