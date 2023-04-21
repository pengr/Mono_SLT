#!/bin/bash
cd ~/mono_slt
# Generate the fake Gloss data of reference by Back-translation Model
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/bt_phoenix/bt \
--path ../checkpoints/bt_phoenix/bt/checkpoint_best.pt --gen-subset train --beam 4 --batch-size 256 \
--results-path ../checkpoints/bt_phoenix/bt

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/bt_phoenix/bt \
--path ../checkpoints/bt_phoenix/bt/checkpoint_best.pt --gen-subset valid --beam 4 --batch-size 256 \
--results-path ../checkpoints/bt_phoenix/bt

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/bt_phoenix/bt \
--path ../checkpoints/bt_phoenix/bt/checkpoint_best.pt --gen-subset test --beam 4 --batch-size 256 \
--results-path ../checkpoints/bt_phoenix/bt

grep ^H ../checkpoints/bt_phoenix/bt/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_phoenix/bt/train.txt
grep ^H ../checkpoints/bt_phoenix/bt/generate-valid.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_phoenix/bt/valid.txt
grep ^H ../checkpoints/bt_phoenix/bt/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_phoenix/bt/test.txt

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/phoenix2014T/bt_phoenix/bt \
# --path ../checkpoints/bt_phoenix/bt/checkpoint_best.pt ../checkpoints/bt_phoenix/bt/checkpoint_best.pt \
# --beam 4 --batch-size 256 --results-path ../checkpoints/bt_phoenix/bt