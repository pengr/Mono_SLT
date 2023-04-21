#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# Multi-domain test datasets
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/mixed.train \
--validpref $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/mixed.valid --testpref $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/${DOMAINS[0]}.test,\
$MULTI_DOMAIN_DIR/gloss_en/bt_mixed/${DOMAINS[1]}.test,$MULTI_DOMAIN_DIR/gloss_en/bt_mixed/${DOMAINS[2]}.test,\
$MULTI_DOMAIN_DIR/gloss_en/bt_mixed/${DOMAINS[3]}.test --destdir $MULTI_DOMAIN_DIR/gloss_en/bt_mixed \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower