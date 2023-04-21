#!/bin/bash
cd rule_alsg_proc3
bash rule_gen.sh &&
bash fthr_pre.sh &&
bash fthr_train.sh &&
bash fthr_gen.sh &&
bash fthr_score.sh