#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain

python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $MULTI_DOMAIN_DIR/gloss_en/bt_ted/ted.train \
--validpref $MULTI_DOMAIN_DIR/gloss_en/bt_ted/ted.valid --testpref $MULTI_DOMAIN_DIR/gloss_en/bt_ted/ted.test \
--destdir $MULTI_DOMAIN_DIR/gloss_en/bt_ted --nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower