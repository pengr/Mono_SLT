#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix3/fthr \
--path ../checkpoints/rule_phoenix3/pre/checkpoint_best.pt --gen-subset valid --beam 4 --batch-size 256 \
--results-path ../checkpoints/rule_phoenix3/pre

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix3/fthr \
--path ../checkpoints/rule_phoenix3/pre/checkpoint_best.pt --gen-subset test --beam 4 --batch-size 256 \
--results-path ../checkpoints/rule_phoenix3/pre

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/rule_phoenix3/fthr \
# --path ../checkpoints/rule_phoenix3/pre/checkpoint_best.pt ../checkpoints/rule_phoenix3/pre/checkpoint_last.pt \
# --beam 4 --batch-size 256 --results-path ../checkpoints/rule_phoenix3/pre