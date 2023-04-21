#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain

python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $MULTI_DOMAIN_DIR/gloss_en/rule_news/news.train \
--validpref $MULTI_DOMAIN_DIR/gloss_en/rule_news/news.valid --testpref $MULTI_DOMAIN_DIR/gloss_en/rule_news/news.test \
--destdir $MULTI_DOMAIN_DIR/gloss_en/rule_news --nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower