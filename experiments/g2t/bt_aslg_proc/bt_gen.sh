#!/bin/bash
cd ~/mono_slt
ASLG_DIR=../data/aslg
WMT14_DIR=../data/wmt14_en_de

# Binarize EN monolingual data of WMT14 dataset
python preprocess_slt.py --only-source --source-lang en --target-lang gloss_proc --trainpref $WMT14_DIR/train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --srcdict $ASLG_DIR/bt_aslg_proc/bt/dict.en.txt \
--destdir $ASLG_DIR/bt_aslg_proc/mono --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# Generate the monolingual Gloss data by Back-translation Model
cp -f $ASLG_DIR/bt_aslg_proc/bt/dict.gloss_proc.txt $ASLG_DIR/bt_aslg_proc/mono/dict.gloss_proc.txt
CUDA_VISIBLE_DEVICES=0 python generate.py $ASLG_DIR/bt_aslg_proc/mono \
--path ../checkpoints/bt_aslg_proc/bt/checkpoint_best.pt \
--gen-subset train --beam 5 --batch-size 256 --results-path $ASLG_DIR/bt_aslg_proc/mono

# Extract the sorted translations from generate-train.txt
mkdir -p $WMT14_DIR/bt
grep ^H $ASLG_DIR/bt_aslg_proc/mono/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> $WMT14_DIR/bt/train.gloss_proc