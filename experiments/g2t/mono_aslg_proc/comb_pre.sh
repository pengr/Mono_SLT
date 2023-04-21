#!/bin/bash
cd ~/mono_slt
ASLG_DIR=../data/aslg
WMT14_DIR=../data/wmt14_en_de

# cat ASLG-PC12 EN-Gloss and WMT14 EN train datasets
rm -rf $ASLG_DIR/train.mono_comb.en $ASLG_DIR/train.mono_comb.gloss_proc
cat $ASLG_DIR/train.en $WMT14_DIR/train.en >> $ASLG_DIR/train.mono_comb.en
cat $ASLG_DIR/train.gloss_proc $WMT14_DIR/train.en >> $ASLG_DIR/train.mono_comb.gloss_proc

# ASLG-PC12 EN-Gloss Pre-Processing of Combined Training
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref $ASLG_DIR/train.mono_comb \
--validpref $ASLG_DIR/dev --testpref $ASLG_DIR/test --destdir $ASLG_DIR/mono_aslg_proc/comb \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# delete the concatenated train datasets
rm -rf $ASLG_DIR/train.mono_comb.en $ASLG_DIR/train.mono_comb.gloss_proc