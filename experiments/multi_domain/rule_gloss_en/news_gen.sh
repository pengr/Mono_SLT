#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_en/rule_news \
--path ../checkpoints/multi_domain/gloss_en/rule_news/checkpoint_best.pt --beam 5 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_en/rule_news

# ensemble decoding
# CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_en/rule_news \
# --path ../checkpoints/multi_domain/gloss_en/rule_news/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_en/rule_news/checkpoint_last.pt --beam 5 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_en/rule_news