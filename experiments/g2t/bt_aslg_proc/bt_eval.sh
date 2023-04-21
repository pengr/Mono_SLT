#!/bin/bash
cd ~/mono_slt
# Generate the fake Gloss data of reference by Back-translation Model
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/bt_aslg_proc/bt \
--path ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt --gen-subset train --beam 5 --batch-size 256 \
--results-path ../checkpoints/bt_aslg_proc/bt

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/bt_aslg_proc/bt \
--path ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt --gen-subset valid --beam 5 --batch-size 256 \
--results-path ../checkpoints/bt_aslg_proc/bt

CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/bt_aslg_proc/bt \
--path ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt --gen-subset test --beam 5 --batch-size 256 \
--results-path ../checkpoints/bt_aslg_proc/bt

grep ^H ../checkpoints/bt_aslg_proc/bt/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_aslg_proc/bt/train.txt
grep ^H ../checkpoints/bt_aslg_proc/bt/generate-valid.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_aslg_proc/bt/valid.txt
grep ^H ../checkpoints/bt_aslg_proc/bt/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/bt_aslg_proc/bt/test.txt

# ensemble decoding
# CUDA_VISIBLE_DEVICES=0 python generate.py ../data/aslg/bt_aslg_proc/bt \
# --path ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt \
# --beam 5 --batch-size 256 --results-path ../checkpoints/bt_aslg_proc/bt