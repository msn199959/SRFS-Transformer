#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 5228 train_distributed.py --gpu_id '0' \
    --gray_aug --gray_p 0.3 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr 1e-4  \
    --batch_size 128 --num_patch 1 --threshold 0.35 --test_per_epoch 2 \
    --dataset jhu --crop_size 256 --pre None --test_patch --save --using_refinement \
    --test_start_epoch 50 --starting_epoch 80 --refine_interval 5 \
    --refine_replace True  --train_number 126 --refine_weight 2.0