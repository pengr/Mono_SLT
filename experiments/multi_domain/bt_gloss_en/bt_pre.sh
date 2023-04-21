#!/bin/bash
cd ~/mono_slt
ASLG_DIR=../data/aslg
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# cat ASLG-PC12 EN-Gloss and Multi-Domain EN train datasets
rm -rf $MULTI_DOMAIN_DIR/mixed.train.en $MULTI_DOMAIN_DIR/mixed.train.gloss_proc
cat $ASLG_DIR/train.en >> $MULTI_DOMAIN_DIR/mixed.train.en
cat $ASLG_DIR/train.gloss_proc >> $MULTI_DOMAIN_DIR/mixed.train.gloss_proc
for domain in "${DOMAINS[@]}"; do
    cat $MULTI_DOMAIN_DIR/$domain.train.en >> $MULTI_DOMAIN_DIR/mixed.train.en
    cat $MULTI_DOMAIN_DIR/$domain.train.en >> $MULTI_DOMAIN_DIR/mixed.train.gloss_proc
done

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang en --target-lang gloss_proc --trainpref $MULTI_DOMAIN_DIR/mixed.train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --destdir $MULTI_DOMAIN_DIR/gloss_en/bt \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $MULTI_DOMAIN_DIR/gloss_en/bt/*.idx $MULTI_DOMAIN_DIR/gloss_en/bt/*.bin
rm -rf $MULTI_DOMAIN_DIR/mixed.train.en $MULTI_DOMAIN_DIR/mixed.train.gloss_proc

# ASLG-PC12 EN-Gloss Pre-Processing of Back-Translation
python preprocess_slt.py --source-lang en --target-lang gloss_proc --trainpref $ASLG_DIR/train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --srcdict $MULTI_DOMAIN_DIR/gloss_en/bt/dict.en.txt \
--tgtdict $MULTI_DOMAIN_DIR/gloss_en/bt/dict.gloss_proc.txt --destdir $MULTI_DOMAIN_DIR/gloss_en/bt \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower