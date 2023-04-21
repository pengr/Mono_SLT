#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix/comb \
--path ../checkpoints/rule_phoenix/comb/checkpoint_best.pt --gen-subset valid --beam 4 --batch-size 256 \
--results-path ../checkpoints/rule_phoenix/comb

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix/comb \
--path ../checkpoints/rule_phoenix/comb/checkpoint_best.pt --gen-subset test --beam 4 --batch-size 256 \
--results-path ../checkpoints/rule_phoenix/comb

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix/comb \
# --path ../checkpoints/rule_phoenix/comb/checkpoint_best.pt ../checkpoints/rule_phoenix/comb/checkpoint_last.pt \
# --beam 4 --batch-size 256 --results-path ../checkpoints/rule_phoenix/comb