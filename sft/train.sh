MODE=$1 # "program", "grammar"
MODEL_NAME=$2  # "meta-llama/Meta-Llama-3-8B-Instruct"

accelerate launch --config_file=sft/configs/multi_gpu.yaml \
    sft/train.py \
    --model_name MODEL_NAME \
    --dataset_name sft/data/program.json \
    --output_dir sft/log/${MODEL_NAME}_program \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 3072 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --attn_implementation "flash_attention_2" \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_r 16 --lora_alpha 16 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps=10 \
    --report_to "wandb" \
    # --gradient_checkpointing \
