#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_en/bt_ted \
--path ../checkpoints/multi_domain/gloss_en/bt_ted/checkpoint_best.pt --beam 5 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_en/bt_ted

# ensemble decoding
# CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_en/bt_ted \
# --path ../checkpoints/multi_domain/gloss_en/bt_ted/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_en/bt_ted/checkpoint_last.pt --beam 5 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_en/bt_ted