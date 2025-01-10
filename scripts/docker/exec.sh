docker run --gpus all \
    -it --rm \
    --env-file .env \
    --shm-size=4g \
    --memory=4g \
    -v ./log:/ggdg/log \
    -v $HOME/.cache/huggingface:/ggdg/.cache/huggingface \
    ggdg bash
