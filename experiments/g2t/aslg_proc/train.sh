#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=0 python train.py ../data/aslg/aslg_proc \
--task translation --arch transformer_slt --share-decoder-input-output-embed --dropout 0.1 \
--attention-dropout 0.1 --encoder-normalize-before --decoder-normalize-before \
--optimizer adam --adam-betas 0.9,0.998 --adam-eps 1e-9 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 8.67e-08 --warmup-updates 8000 --lr 2e-04 --min-lr 1e-09 \
--weight-decay 0.0 --max-tokens 2048 --save-dir ../checkpoints/aslg_proc --update-freq 3 --no-progress-bar --log-format json \
--log-interval 50 --save-interval-updates 1000 --keep-interval-updates 1 --max-update 100000 \
--patience 5

#ensemble decoding: --keep-interval-updates 5
#only save checkpoint_best.pt: --no-epoch-checkpoints --no-last-checkpoints