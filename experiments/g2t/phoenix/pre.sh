#!/bin/bash
cd ~/mono_slt
python preprocess_slt.py --source-lang gloss --target-lang de --trainpref ../data/phoenix2014T/train \
--validpref ../data/phoenix2014T/dev --testpref ../data/phoenix2014T/test \
--destdir ../data/phoenix2014T/phoenix --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower