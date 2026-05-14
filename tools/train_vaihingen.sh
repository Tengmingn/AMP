#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/vaihingen.yaml
save_path=exp/vaihingen/dinov2b_full_new260210

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train_vaihingen.py \
    --config=$config\
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt