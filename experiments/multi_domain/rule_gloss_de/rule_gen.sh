#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# Move DE monolingual text into each domain (train/valid/test)
for domain in "${DOMAINS[@]}"; do
    mkdir -p $MULTI_DOMAIN_DIR/gloss_de/rule_$domain
    for f in train valid test; do
        cp -f $MULTI_DOMAIN_DIR/$domain.$f.de $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.$f.de
        # Generate the monolingual Gloss data by Transformation Rule (Train,Valid,Test)
        CUDA_VISIBLE_DEVICES=0 python de2gloss.py -input $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.$f.de -output $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.$f.gloss \
        -de_dict $PHOENIX_DIR/phoenix/dict.de.txt -gloss_dict $PHOENIX_DIR/phoenix/dict.gloss.txt
    done
done

mkdir -p $MULTI_DOMAIN_DIR/gloss_de/rule_mixed
for domain in "${DOMAINS[@]}"; do
    # Combined all DE and Gloss text into Mixed domain (train/valid)
    for f in train valid; do
        cat $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.$f.de >> $MULTI_DOMAIN_DIR/gloss_de/rule_mixed/mixed.$f.de
        cat $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.$f.gloss >> $MULTI_DOMAIN_DIR/gloss_de/rule_mixed/mixed.$f.gloss
    done
    # Move all DE and Gloss text into Mixed domain (test)
    cp -f $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.test.de $MULTI_DOMAIN_DIR/gloss_de/rule_mixed/$domain.test.de
    cp -f $MULTI_DOMAIN_DIR/gloss_de/rule_$domain/$domain.test.gloss $MULTI_DOMAIN_DIR/gloss_de/rule_mixed/$domain.test.gloss
done