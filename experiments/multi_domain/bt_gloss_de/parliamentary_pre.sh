#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain

python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $MULTI_DOMAIN_DIR/gloss_de/bt_parliamentary/parliamentary.train \
--validpref $MULTI_DOMAIN_DIR/gloss_de/bt_parliamentary/parliamentary.valid --testpref $MULTI_DOMAIN_DIR/gloss_de/bt_parliamentary/parliamentary.test \
--destdir $MULTI_DOMAIN_DIR/gloss_de/bt_parliamentary --nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower