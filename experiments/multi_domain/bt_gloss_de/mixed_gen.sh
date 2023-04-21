#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/bt_mixed \
--path ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_best.pt --gen-subset test --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/bt_mixed

CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/bt_mixed \
--path ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_best.pt --gen-subset test1 --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/bt_mixed

CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/bt_mixed \
--path ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_best.pt --gen-subset test2 --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/bt_mixed

CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/bt_mixed \
--path ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_best.pt --gen-subset test3 --beam 4 --batch-size 256 \
--results-path ../checkpoints/multi_domain/gloss_de/bt_mixed

# ensemble decoding
# CUDA_VISIBLE_DEVICES=5 python generate.py ../data/multi_domain/gloss_de/bt_mixed \
# --path ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_best.pt \
# ../checkpoints/multi_domain/gloss_de/bt_mixed/checkpoint_last.pt --beam 4 --batch-size 256 \
# --results-path ../checkpoints/multi_domain/gloss_de/bt_mixed