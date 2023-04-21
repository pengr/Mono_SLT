#!/bin/bash
cd ~/mono_slt
ASLG_DIR=../data/aslg
WMT14_DIR=../data/wmt14_en_de

# cat ASLG-PC12 EN-Gloss and WMT14 EN train datasets
rm -rf $ASLG_DIR/train.bt_bt.en $ASLG_DIR/train.bt_bt.gloss_proc
cat $ASLG_DIR/train.en $WMT14_DIR/train.en >> $ASLG_DIR/train.bt_bt.en
cat $ASLG_DIR/train.gloss_proc $WMT14_DIR/train.en >> $ASLG_DIR/train.bt_bt.gloss_proc

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang en --target-lang gloss_proc --trainpref $ASLG_DIR/train.bt_bt \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --destdir $ASLG_DIR/bt_aslg_proc/bt \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $ASLG_DIR/bt_aslg_proc/bt/*.idx $ASLG_DIR/bt_aslg_proc/bt/*.bin
rm -rf $ASLG_DIR/train.bt_bt.en $ASLG_DIR/train.bt_bt.gloss_proc

# ASLG-PC12 EN-Gloss Pre-Processing of Back-Translation
python preprocess_slt.py --source-lang en --target-lang gloss_proc --trainpref $ASLG_DIR/train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --srcdict $ASLG_DIR/bt_aslg_proc/bt/dict.en.txt \
--tgtdict $ASLG_DIR/bt_aslg_proc/bt/dict.gloss_proc.txt --destdir $ASLG_DIR/bt_aslg_proc/bt \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower