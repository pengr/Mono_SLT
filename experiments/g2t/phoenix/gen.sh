#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=4 python generate.py ../data/phoenix2014T/phoenix \
--path ../checkpoints/phoenix/checkpoint_best.pt --gen-subset valid --beam 4 --batch-size 256 \
--results-path ../checkpoints/phoenix

CUDA_VISIBLE_DEVICES=4 python generate.py ../data/phoenix2014T/phoenix \
--path ../checkpoints/phoenix/checkpoint_best.pt --gen-subset test --beam 4 --batch-size 256 \
--results-path ../checkpoints/phoenix

# ensemble decoding
# CUDA_VISIBLE_DEVICES=4 python generate.py ../data/phoenix2014T/phoenix \
# --path ../checkpoints/phoenix/checkpoint_best.pt ../checkpoints/phoenix/checkpoint_last.pt --beam 4 --batch-size 256 \
# --results-path ../checkpoints/phoenix