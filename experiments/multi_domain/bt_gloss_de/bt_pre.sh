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

# cat Phoenix-Weather2014T DE-Gloss and Multi-Domain DE train datasets
rm -rf $MULTI_DOMAIN_DIR/mixed.train.de $MULTI_DOMAIN_DIR/mixed.train.gloss
cat $PHOENIX_DIR/train.de >> $MULTI_DOMAIN_DIR/mixed.train.de
cat $PHOENIX_DIR/train.gloss >> $MULTI_DOMAIN_DIR/mixed.train.gloss
for domain in "${DOMAINS[@]}"; do
    cat $MULTI_DOMAIN_DIR/$domain.train.de >> $MULTI_DOMAIN_DIR/mixed.train.de
    cat $MULTI_DOMAIN_DIR/$domain.train.de >> $MULTI_DOMAIN_DIR/mixed.train.gloss
done

# create the concatenated dictionary, vocab size truncated as 80,000
python preprocess_slt.py --source-lang de --target-lang gloss --trainpref $MULTI_DOMAIN_DIR/mixed.train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --destdir $MULTI_DOMAIN_DIR/gloss_de/bt \
--nwordssrc 80000 --nwordstgt 80000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $MULTI_DOMAIN_DIR/gloss_de/bt/*.idx $MULTI_DOMAIN_DIR/gloss_de/bt/*.bin
rm -rf $MULTI_DOMAIN_DIR/mixed.train.de $MULTI_DOMAIN_DIR/mixed.train.gloss

# merge phoenix dictonary into the concatenated dictionary
python concat_dict.py $MULTI_DOMAIN_DIR/gloss_de/bt/dict.de.txt $PHOENIX_DIR/phoenix/dict.de.txt
python concat_dict.py $MULTI_DOMAIN_DIR/gloss_de/bt/dict.gloss.txt $PHOENIX_DIR/phoenix/dict.gloss.txt

# Phoenix-Weather2014T DE-Gloss Pre-Processing of Back-Translation
python preprocess_slt.py --source-lang de --target-lang gloss --trainpref $PHOENIX_DIR/train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $MULTI_DOMAIN_DIR/gloss_de/bt/dict.de.txt \
--tgtdict $MULTI_DOMAIN_DIR/gloss_de/bt/dict.gloss.txt --destdir $MULTI_DOMAIN_DIR/gloss_de/bt \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower