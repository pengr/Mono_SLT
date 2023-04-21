#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
MULTI_DOMAIN_DIR=../data/multi_domain
DOMAINS=(
    "news"
    "ted"
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)

# Traverse each domain
for domain in "${DOMAINS[@]}"; do
    # Binarize the monolingual DE data of each domain
    python preprocess_slt.py --only-source --source-lang de --target-lang gloss --trainpref $MULTI_DOMAIN_DIR/$domain.train \
    --validpref $MULTI_DOMAIN_DIR/$domain.valid --testpref $MULTI_DOMAIN_DIR/$domain.test --srcdict $MULTI_DOMAIN_DIR/gloss_de/bt/dict.de.txt \
    --destdir $MULTI_DOMAIN_DIR/gloss_de/mono_$domain --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

    # Generate the monolingual Gloss data by Back-translation Model
    cp -f $MULTI_DOMAIN_DIR/gloss_de/bt/dict.gloss.txt $MULTI_DOMAIN_DIR/gloss_de/mono_$domain/dict.gloss.txt
    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_de/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_de/bt/checkpoint_best.pt \
    --gen-subset train --beam 4 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_de/mono_$domain

    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_de/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_de/bt/checkpoint_best.pt \
    --gen-subset valid --beam 4 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_de/mono_$domain

    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_de/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_de/bt/checkpoint_best.pt \
    --gen-subset test --beam 4 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_de/mono_$domain

    # Extract the sorted translations from generate-train/valid/test.txt
    mkdir -p $MULTI_DOMAIN_DIR/gloss_de/bt_$domain
    grep ^H $MULTI_DOMAIN_DIR/gloss_de/mono_$domain/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.train.gloss
    grep ^H $MULTI_DOMAIN_DIR/gloss_de/mono_$domain/generate-valid.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.valid.gloss
    grep ^H $MULTI_DOMAIN_DIR/gloss_de/mono_$domain/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_de/bt_$domain/$domain.test.gloss
done
