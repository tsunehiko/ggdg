SEED=0

python ggdg/main.py \
    --seed $SEED \
    --exp_name main/ggdg/${SEED} \
    --dataset ludii \
    --engine hf/meta-llama/Llama-3-8B-Instruct \
    --tokenizer meta-llama/Llama-3-8B-Instruct \
    --output_max_tokens 2048 \
    --input_max_tokens 8192 \
    --prompt_mode grammar \
    --game_list_path "data/ludii/game_dicts/main.json" \
    --test_num 100 \
    --constrain_rule_gen_flag \
    --constrain_prog_gen_flag \
    --gd_max_num_correction 10 \

    # --random_sampling \
    # --use_oracle_rule_flag \
