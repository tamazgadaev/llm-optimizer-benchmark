#!/bin/bash

# NOTE on DDP scaling in this repo:
# - `iterations` counts OPTIMIZER STEPS (see `src/optim/base.py`: curr_iter increments after `opt.step()`).
# - The DDP backend *rewrites* `--batch_size` and `--acc_steps` per-rank (see `src/distributed/ddp.py`)
#   to keep `effective_batch_size = batch_size * acc_steps` constant globally across `world_size`.
# - So to keep the *token budget per optimizer step* constant when changing `--nproc_per_node`,
#   keep `batch_size * acc_steps` constant.
#
# Global tokens / optimizer-step = (batch_size * acc_steps) * sequence_length

torchrun --nproc_per_node=8 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 128 --sequence_length 512 --acc_steps 2 \
    --dataset fineweb --iterations 16010 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt mars --lr 1e-3 --mars_lr 3e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 \
    --wandb --wandb_project mars_understanding  --wandb_entity tamaz \
    --eval_interval 200 --latest_ckpt_interval 1000 \
    --log_timestamps --auto_resume False \