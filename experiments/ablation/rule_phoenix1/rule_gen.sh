#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# Generate the wmt14 Gloss data by Transformation Rule (Train)
mkdir -p $WMT14_DIR/rule1
cp -f $WMT14_DIR/train.de $WMT14_DIR/rule1/train.de
CUDA_VISIBLE_DEVICES=0 python de2gloss.py -input  $WMT14_DIR/rule1/train.de -output  $WMT14_DIR/rule1/train.gloss \
-de_dict $PHOENIX_DIR/phoenix/dict.de.txt -gloss_dict $PHOENIX_DIR/phoenix/dict.gloss.txt -no_handle_punct_spec