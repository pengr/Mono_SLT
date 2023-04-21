#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/bt_aslg_proc/comb \
--path ../checkpoints/bt_aslg_proc/comb/checkpoint_best.pt --gen-subset valid --beam 5 --batch-size 256 \
--results-path ../checkpoints/bt_aslg_proc/comb

CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/bt_aslg_proc/comb \
--path ../checkpoints/bt_aslg_proc/comb/checkpoint_best.pt --gen-subset test --beam 5 --batch-size 256 \
--results-path ../checkpoints/bt_aslg_proc/comb

# ensemble decoding
# CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/bt_aslg_proc/comb \
# --path ../checkpoints/bt_aslg_proc/comb/checkpoint_best.pt ../checkpoints/bt_aslg_proc/comb/checkpoint_last.pt \
# --beam 5 --batch-size 256 --results-path ../checkpoints/bt_aslg_proc/comb