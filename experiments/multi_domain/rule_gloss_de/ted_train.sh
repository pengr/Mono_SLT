#!/bin/bash
cd ~/mono_slt
CUDA_VISIBLE_DEVICES=5 python train.py ../data/multi_domain/gloss_de/rule_ted \
--task translation --arch transformer_slt --dropout 0.1 --attention-dropout 0.1 --encoder-normalize-before --decoder-normalize-before \
--optimizer adam --adam-betas 0.9,0.998 --adam-eps 1e-9 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 2.74e-07 --warmup-updates 3000 --lr 5.5e-04 --min-lr 1e-09 \
--weight-decay 0.0 --max-tokens 2048 --save-dir ../checkpoints/multi_domain/gloss_de/rule_ted --update-freq 3 --no-progress-bar \
--log-format json --log-interval 50 --save-interval-updates 100 --keep-interval-updates 1 --max-update 100000 \
--patience 3

#ensemble decoding: --keep-interval-updates 9
#only save checkpoint_best.pt: --no-epoch-checkpoints --no-last-checkpoints