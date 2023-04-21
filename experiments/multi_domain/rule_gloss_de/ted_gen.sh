#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/rule_ted \
--path ../checkpoints/multi_domain/gloss_de/rule_ted/checkpoint_best.pt --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/rule_ted

# ensemble decoding
# CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/rule_ted \
# --path ../checkpoints/multi_domain/gloss_de/rule_ted/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_de/rule_ted/checkpoint_last.pt --beam 4 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_de/rule_ted