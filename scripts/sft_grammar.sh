BASE_MODEL_NAME=$1 # "meta-llama/Meta-Llama-3-8B-Instruct"
SEED=0

python ggdg/main.py \
    --seed $SEED \
    --exp_name main/debug/${SEED} \
    --dataset ludii \
    --prompt_mode grammar \
    --game_list_path "data/ludii/game_dicts/main.json" \
    --test_num 100 \
    --num_shot 0 \
    --engine local/sft/log/${BASE_MODEL_NAME}_program \
    --tokenizer $BASE_MODEL_NAME \
    --output_max_tokens 2048 \
    --input_max_tokens 8192 \
    --terminal_first \
    --use_oracle_rule_flag \
    --gd_max_num_correction 30 \
    --constrain_prog_gen_flag \
    --constrain_rule_gen_flag \
    --grammar_engine local/sft/log/${BASE_MODEL_NAME}_grammar \
    --grammar_tokenizer $BASE_MODEL_NAME \
    --grammar_output_max_tokens 2048 \
    --grammar_input_max_tokens 8192 \

