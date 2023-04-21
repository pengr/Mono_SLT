#!/bin/bash
cd ~/mono_slt
python preprocess_slt.py --source-lang gloss_proc --target-lang en --trainpref ../data/aslg/train \
--validpref ../data/aslg/dev --testpref ../data/aslg/test \
--destdir ../data/aslg/aslg_proc --thresholdtgt 0 --thresholdsrc 0 --workers 16 --lower