#!/bin/bash
cd ~/mono_slt
ASLG_DIR=../data/aslg
WMT14_DIR=../data/wmt14_en_de

# cat ASLG-PC12 EN-Gloss and WMT14 EN train datasets
rm -rf $ASLG_DIR/train.bt_fthr.en $ASLG_DIR/train.bt_fthr.gloss_proc
cp -f $WMT14_DIR/train.en $WMT14_DIR/bt/train.en
cat $ASLG_DIR/train.en $WMT14_DIR/bt/train.en >> $ASLG_DIR/train.bt_fthr.en
cat $ASLG_DIR/train.gloss_proc $WMT14_DIR/bt/train.gloss_proc >> $ASLG_DIR/train.bt_fthr.gloss_proc

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $ASLG_DIR/train.bt_fthr \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --destdir $ASLG_DIR/bt_aslg_proc/pre \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $ASLG_DIR/bt_aslg_proc/pre/*.idx $ASLG_DIR/bt_aslg_proc/pre/*.bin
rm -rf $ASLG_DIR/train.bt_fthr.en $ASLG_DIR/train.bt_fthr.gloss_proc

# ASLG-PC12 EN-Gloss Pre-Processing of Pre-Training(use WMT14 EN-Gloss)
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $WMT14_DIR/bt/train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --srcdict $ASLG_DIR/bt_aslg_proc/pre/dict.gloss_proc.txt \
--tgtdict $ASLG_DIR/bt_aslg_proc/pre/dict.en.txt --destdir $ASLG_DIR/bt_aslg_proc/pre \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# ASLG-PC12 EN-Gloss Pre-Processing of Further Training(use WMT14 EN-Gloss)
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $ASLG_DIR/train \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --srcdict $ASLG_DIR/bt_aslg_proc/pre/dict.gloss_proc.txt \
--tgtdict $ASLG_DIR/bt_aslg_proc/pre/dict.en.txt --destdir $ASLG_DIR/bt_aslg_proc/fthr \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower