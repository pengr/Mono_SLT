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

# Move EN monolingual text into each domain (train/valid/test)
for domain in "${DOMAINS[@]}"; do
    mkdir -p $MULTI_DOMAIN_DIR/gloss_en/rule_$domain
    for f in train valid test; do
        cp -f $MULTI_DOMAIN_DIR/$domain.$f.en $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.$f.en
        # Generate the monolingual Gloss data by Transformation Rule (Train,Valid,Test)
        CUDA_VISIBLE_DEVICES=0 python en2gloss.py -input $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.$f.en \
        -output $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.$f.gloss_proc
    done
done

mkdir -p $MULTI_DOMAIN_DIR/gloss_en/rule_mixed
for domain in "${DOMAINS[@]}"; do
    # Combined all EN and Gloss text into Mixed domain (train/valid)
    for f in train valid; do
        cat $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.$f.en >> $MULTI_DOMAIN_DIR/gloss_en/rule_mixed/mixed.$f.en
        cat $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.$f.gloss_proc >> $MULTI_DOMAIN_DIR/gloss_en/rule_mixed/mixed.$f.gloss_proc
    done
    # Move all EN and Gloss text into Mixed domain (test)
    cp -f $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.test.en $MULTI_DOMAIN_DIR/gloss_en/rule_mixed/$domain.test.en
    cp -f $MULTI_DOMAIN_DIR/gloss_en/rule_$domain/$domain.test.gloss_proc $MULTI_DOMAIN_DIR/gloss_en/rule_mixed/$domain.test.gloss_proc
done