#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# Binarize DE monolingual data of WMT14 dataset
python preprocess_slt.py --only-source --source-lang de --target-lang gloss --trainpref $WMT14_DIR/train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $PHOENIX_DIR/bt_phoenix/bt/dict.de.txt \
--destdir $PHOENIX_DIR/bt_phoenix/mono --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# Generate the monolingual Gloss data by Back-translation Model
cp -f $PHOENIX_DIR/bt_phoenix/bt/dict.gloss.txt $PHOENIX_DIR/bt_phoenix/mono/dict.gloss.txt
CUDA_VISIBLE_DEVICES=0 python generate.py $PHOENIX_DIR/bt_phoenix/mono \
--path ../checkpoints/bt_phoenix/bt/checkpoint_best.pt \
--gen-subset train --beam 4 --batch-size 256 --results-path $PHOENIX_DIR/bt_phoenix/mono

# Extract the sorted translations from generate-train.txt
mkdir -p $WMT14_DIR/bt
grep ^H $PHOENIX_DIR/bt_phoenix/mono/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> $WMT14_DIR/bt/train.gloss
