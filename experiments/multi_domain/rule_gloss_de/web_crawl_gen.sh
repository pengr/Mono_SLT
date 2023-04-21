#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/rule_web_crawl \
--path ../checkpoints/multi_domain/gloss_de/rule_web_crawl/checkpoint_best.pt --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/rule_web_crawl

# ensemble decoding
# CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/rule_web_crawl \
# --path ../checkpoints/multi_domain/gloss_de/rule_web_crawl/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_de/rule_web_crawl/checkpoint_last.pt --beam 4 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_de/rule_web_crawl