#!/bin/bash
cd ~/mono_slt
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# Move DE monolingual text into each domain except Mixed domain (train/valid/test)
for f in train.de valid.de test.de; do
    for domain in "${DOMAINS[@]}"; do
        cp -f $MULTI_DOMAIN_DIR/$domain.$f $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.$f
    done
done

# Combined all DE text into Mixed domain (train/valid)
mkdir -p $MULTI_DOMAIN_DIR/gloss_de/bt_mixed
for f in train.de valid.de; do
    for domain in "${DOMAINS[@]}"; do
        cat $MULTI_DOMAIN_DIR/$domain.$f >> $MULTI_DOMAIN_DIR/gloss_de/bt_mixed/mixed.$f
    done
done
# Combined all Gloss text into Mixed domain (train/valid)
for f in train.gloss valid.gloss; do
    for domain in "${DOMAINS[@]}"; do
        cat $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.$f >> $MULTI_DOMAIN_DIR/gloss_de/bt_mixed/mixed.$f
    done
done
# Move all DE and Gloss text into Mixed domain (test)
for domain in "${DOMAINS[@]}"; do
    cp -f $MULTI_DOMAIN_DIR/$domain.test.de $MULTI_DOMAIN_DIR/gloss_de/bt_mixed/$domain.test.de
    cp -f $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.test.gloss $MULTI_DOMAIN_DIR/gloss_de/bt_mixed/$domain.test.gloss
done