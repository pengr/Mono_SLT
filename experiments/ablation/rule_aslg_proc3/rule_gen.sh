#!/bin/bash
cd ~/mono_slt
# Generate the wmt14 Gloss data by Transformation Rule (Train)
mkdir -p ../data/wmt14_en_de/rule3
CUDA_VISIBLE_DEVICES=0 python en2gloss.py -input ../data/wmt14_en_de/train.en -output ../data/wmt14_en_de/rule3/train.gloss_proc -no_handle_punct_spec