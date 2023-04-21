#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/multi_domain/gloss_de/bt_news \
--path ../checkpoints/multi_domain/gloss_de/bt_news/checkpoint_best.pt --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/bt_news

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/multi_domain/gloss_de/bt_news \
# --path ../checkpoints/multi_domain/gloss_de/bt_news/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_de/bt_news/checkpoint_last.pt --beam 4 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_de/bt_news