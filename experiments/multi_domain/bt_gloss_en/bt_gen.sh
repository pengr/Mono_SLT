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

# Traverse each domain
for domain in "${DOMAINS[@]}"; do
    # Binarize the monolingual EN data of each domain
    python preprocess_slt.py --only-source --source-lang en --target-lang gloss_proc --trainpref $MULTI_DOMAIN_DIR/$domain.train \
    --validpref $MULTI_DOMAIN_DIR/$domain.valid --testpref $MULTI_DOMAIN_DIR/$domain.test --srcdict $MULTI_DOMAIN_DIR/gloss_en/bt/dict.en.txt \
    --destdir $MULTI_DOMAIN_DIR/gloss_en/mono_$domain --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

    # Generate the monolingual Gloss data by Back-translation Model
    cp -f $MULTI_DOMAIN_DIR/gloss_en/bt/dict.gloss_proc.txt $MULTI_DOMAIN_DIR/gloss_en/mono_$domain/dict.gloss_proc.txt
    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_en/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_en/bt/checkpoint_best.pt \
    --gen-subset train --beam 5 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_en/mono_$domain

    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_en/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_en/bt/checkpoint_best.pt \
    --gen-subset valid --beam 5 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_en/mono_$domain

    CUDA_VISIBLE_DEVICES=0 python generate.py $MULTI_DOMAIN_DIR/gloss_en/mono_$domain \
    --path ../checkpoints/multi_domain/gloss_en/bt/checkpoint_best.pt \
    --gen-subset test --beam 5 --batch-size 256 --results-path $MULTI_DOMAIN_DIR/gloss_en/mono_$domain

    # Extract the sorted translations from generate-train/valid/test.txt
    mkdir -p $MULTI_DOMAIN_DIR/gloss_en/bt_$domain
    grep ^H $MULTI_DOMAIN_DIR/gloss_en/mono_$domain/generate-train.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.train.gloss_proc
    grep ^H $MULTI_DOMAIN_DIR/gloss_en/mono_$domain/generate-valid.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.valid.gloss_proc
    grep ^H $MULTI_DOMAIN_DIR/gloss_en/mono_$domain/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 >> $MULTI_DOMAIN_DIR/gloss_en/bt_$domain/$domain.test.gloss_proc
done
