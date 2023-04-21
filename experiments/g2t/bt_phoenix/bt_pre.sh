#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# cat Phoenix-Weather 2014T DE-Gloss and WMT14 DE train datasets
rm -rf $PHOENIX_DIR/train.bt_bt.de $PHOENIX_DIR/train.bt_bt.gloss
cat $PHOENIX_DIR/train.de $WMT14_DIR/train.de >> $PHOENIX_DIR/train.bt_bt.de
cat $PHOENIX_DIR/train.gloss $WMT14_DIR/train.de >> $PHOENIX_DIR/train.bt_bt.gloss

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang de --target-lang gloss --trainpref $PHOENIX_DIR/train.bt_bt \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --destdir $PHOENIX_DIR/bt_phoenix/bt \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $PHOENIX_DIR/bt_phoenix/bt/*.idx $PHOENIX_DIR/bt_phoenix/bt/*.bin
rm -rf $PHOENIX_DIR/train.bt_bt.de $PHOENIX_DIR/train.bt_bt.gloss

# merge phoenix dictonary into the concatenated dictionary
python concat_dict.py $PHOENIX_DIR/bt_phoenix/bt/dict.de.txt $PHOENIX_DIR/phoenix/dict.de.txt
python concat_dict.py $PHOENIX_DIR/bt_phoenix/bt/dict.gloss.txt $PHOENIX_DIR/phoenix/dict.gloss.txt

# Phoenix-Weather 2014T DE-Gloss Pre-Processing of Back-Translation
python preprocess_slt.py --source-lang de --target-lang gloss --trainpref $PHOENIX_DIR/train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $PHOENIX_DIR/bt_phoenix/bt/dict.de.txt \
--tgtdict $PHOENIX_DIR/bt_phoenix/bt/dict.gloss.txt --destdir $PHOENIX_DIR/bt_phoenix/bt \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower