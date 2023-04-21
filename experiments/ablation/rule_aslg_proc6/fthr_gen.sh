#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/rule_aslg_proc6/fthr \
--path ../checkpoints/rule_aslg_proc6/pre/checkpoint_best.pt --gen-subset valid --beam 5 --batch-size 256 \
--results-path ../checkpoints/rule_aslg_proc6/pre

CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/rule_aslg_proc6/fthr \
--path ../checkpoints/rule_aslg_proc6/pre/checkpoint_best.pt --gen-subset test --beam 5 --batch-size 256 \
--results-path ../checkpoints/rule_aslg_proc6/pre

# ensemble decoding
# CUDA_VISIBLE_DEVICES=4 python generate.py ../data/aslg/rule_aslg_proc6/fthr \
# --path ../checkpoints/rule_aslg_proc6/pre/checkpoint_best.pt ../checkpoints/rule_aslg_proc6/pre/checkpoint_last.pt \
# --beam 5 --batch-size 256 --results-path ../checkpoints/rule_aslg_proc6/pre