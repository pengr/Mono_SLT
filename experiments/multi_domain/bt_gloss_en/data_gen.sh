#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# Move EN monolingual text into each domain except Mixed domain (train/valid/test)
for f in train.en valid.en test.en; do
    for domain in "${DOMAINS[@]}"; do
        cp -f $MULTI_DOMAIN_DIR/$domain.$f $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.$f
    done
done

# Combined all EN text into Mixed domain (train/valid)
mkdir -p $MULTI_DOMAIN_DIR/gloss_en/bt_mixed
for f in train.en valid.en; do
    for domain in "${DOMAINS[@]}"; do
        cat $MULTI_DOMAIN_DIR/$domain.$f >> $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/mixed.$f
    done
done
# Combined all Gloss text into Mixed domain (train/valid)
for f in train.gloss_proc valid.gloss_proc; do
    for domain in "${DOMAINS[@]}"; do
        cat $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.$f >> $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/mixed.$f
    done
done
# Move all EN and Gloss text into Mixed domain (test)
for domain in "${DOMAINS[@]}"; do
    cp -f $MULTI_DOMAIN_DIR/$domain.test.en $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/$domain.test.en
    cp -f $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.test.gloss_proc $MULTI_DOMAIN_DIR/gloss_en/bt_mixed/$domain.test.gloss_proc
done