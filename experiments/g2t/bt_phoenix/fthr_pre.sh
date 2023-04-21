#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# cat Phoenix-Weather 2014T DE-Gloss and WMT14 DE train datasets
rm -rf $PHOENIX_DIR/train.bt_fthr.de $PHOENIX_DIR/train.bt_fthr.gloss
cp -f $WMT14_DIR/train.de $WMT14_DIR/bt/train.de
cat $PHOENIX_DIR/train.de $WMT14_DIR/bt/train.de >> $PHOENIX_DIR/train.bt_fthr.de
cat $PHOENIX_DIR/train.gloss $WMT14_DIR/bt/train.gloss >> $PHOENIX_DIR/train.bt_fthr.gloss

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $PHOENIX_DIR/train.bt_fthr \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --destdir $PHOENIX_DIR/bt_phoenix/pre \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $PHOENIX_DIR/bt_phoenix/pre/*.idx $PHOENIX_DIR/bt_phoenix/pre/*.bin
rm -rf $PHOENIX_DIR/train.bt_fthr.de $PHOENIX_DIR/train.bt_fthr.gloss

# merge phoenix dictonary into the concatenated dictionary
python concat_dict.py $PHOENIX_DIR/bt_phoenix/pre/dict.de.txt $PHOENIX_DIR/phoenix/dict.de.txt
python concat_dict.py $PHOENIX_DIR/bt_phoenix/pre/dict.gloss.txt $PHOENIX_DIR/phoenix/dict.gloss.txt

# Phoenix-Weather 2014T DE-Gloss Pre-Processing of Pre-Training(use WMT14 DE-Gloss)
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $WMT14_DIR/bt/train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $PHOENIX_DIR/bt_phoenix/pre/dict.gloss.txt \
--tgtdict $PHOENIX_DIR/bt_phoenix/pre/dict.de.txt --destdir $PHOENIX_DIR/bt_phoenix/pre \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# Phoenix-Weather 2014T DE-Gloss Pre-Processing of Further Training(use WMT14 DE-Gloss)
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $PHOENIX_DIR/train \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $PHOENIX_DIR/bt_phoenix/pre/dict.gloss.txt \
--tgtdict $PHOENIX_DIR/bt_phoenix/pre/dict.de.txt --destdir $PHOENIX_DIR/bt_phoenix/fthr \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower