#!/bin/bash
cd ~/mono_slt

# ASLG-PC12 Gloss-EN Pre-Training(set 300K training steps)
CUDA_VISIBLE_DEVICES=1 python train.py ../data/aslg/rule_aslg_proc2/pre \
--task translation --arch transformer_slt --share-decoder-input-output-embed --dropout 0.1 \
--attention-dropout 0.1 --encoder-normalize-before --decoder-normalize-before \
--optimizer adam --adam-betas 0.9,0.998 --adam-eps 1e-9 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 8.67e-08 --warmup-updates 8000 --lr 2e-04 --min-lr 1e-09 \
--weight-decay 0.0 --max-tokens 2048 --save-dir ../checkpoints/rule_aslg_proc2/pre --update-freq 3 --no-progress-bar --log-format json \
--log-interval 50 --save-interval-updates 1000 --keep-interval-updates 1 --max-update 300000 \
--patience 5

# ASLG-PC12 Gloss-EN Further Training(set 400K training steps, i.e. futher train 100K afther 300K traing steps)
CUDA_VISIBLE_DEVICES=1 python train.py ../data/aslg/rule_aslg_proc2/fthr \
--task translation --arch transformer_slt --share-decoder-input-output-embed --dropout 0.1 \
--attention-dropout 0.1 --encoder-normalize-before --decoder-normalize-before \
--optimizer adam --adam-betas 0.9,0.998 --adam-eps 1e-9 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 8.67e-08 --warmup-updates 8000 --lr 2e-04 --min-lr 1e-09 \
--weight-decay 0.0 --max-tokens 2048 --save-dir ../checkpoints/rule_aslg_proc2/pre --update-freq 3 --no-progress-bar --log-format json \
--log-interval 50 --save-interval-updates 1000 --keep-interval-updates 1 --max-update 400000 \
--patience 5

#ensemble decoding: --keep-interval-updates 5
#only save checkpoint_best.pt: --no-epoch-checkpoints --no-last-checkpoints