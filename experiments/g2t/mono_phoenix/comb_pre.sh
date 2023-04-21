#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# cat Phoenix-Weather 2014T DE-Gloss and WMT14 DE train datasets
rm -rf $PHOENIX_DIR/train.mono_comb.de $PHOENIX_DIR/train.mono_comb.gloss
cat $PHOENIX_DIR/train.de $WMT14_DIR/train.de >> $PHOENIX_DIR/train.mono_comb.de
cat $PHOENIX_DIR/train.gloss $WMT14_DIR/train.de >> $PHOENIX_DIR/train.mono_comb.gloss

# create the concatenated dictionary, vocab size truncated as 50,000
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $PHOENIX_DIR/train.mono_comb \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --destdir $PHOENIX_DIR/mono_phoenix/comb \
--nwordssrc 50000 --nwordstgt 50000 --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# only reserve the concatenated dictionary
rm -f $PHOENIX_DIR/mono_phoenix/comb/*.idx $PHOENIX_DIR/mono_phoenix/comb/*.bin

# merge phoenix dictonary into the concatenated dictionary
python concat_dict.py $PHOENIX_DIR/mono_phoenix/comb/dict.de.txt $PHOENIX_DIR/phoenix/dict.de.txt
python concat_dict.py $PHOENIX_DIR/mono_phoenix/comb/dict.gloss.txt $PHOENIX_DIR/phoenix/dict.gloss.txt

# Phoenix-Weather 2014T DE-Gloss Pre-Processing of Combined Training
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref $PHOENIX_DIR/train.mono_comb \
--validpref $PHOENIX_DIR/dev --testpref $PHOENIX_DIR/test --srcdict $PHOENIX_DIR/mono_phoenix/comb/dict.gloss.txt \
--tgtdict $PHOENIX_DIR/mono_phoenix/comb/dict.de.txt --destdir $PHOENIX_DIR/mono_phoenix/comb \
--thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower

# delete the concatenated train datasets
rm -rf $PHOENIX_DIR/train.mono_comb.de $PHOENIX_DIR/train.mono_comb.gloss