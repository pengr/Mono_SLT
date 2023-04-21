#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/aslg_proc \
--path ../checkpoints/aslg_proc/checkpoint_best.pt --gen-subset valid --beam 5 --batch-size 256 \
--results-path ../checkpoints/aslg_proc

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/aslg_proc \
--path ../checkpoints/aslg_proc/checkpoint_best.pt --gen-subset test --beam 5 --batch-size 256 \
--results-path ../checkpoints/aslg_proc

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/aslg_proc \
# --path ../checkpoints/aslg_proc/checkpoint_best.pt ../checkpoints/aslg_proc/checkpoint_best.pt --beam 5 --batch-size 256 \
# --results-path ../checkpoints/aslg_proc