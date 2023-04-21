#!/bin/bash
cd rule_gloss_en
bash rule_gen.sh &&
bash news_pre.sh &&
bash news_train.sh &&
bash news_gen.sh &&
bash parliamentary_pre.sh &&
bash parliamentary_train.sh &&
bash parliamentary_gen.sh &&
bash ted_pre.sh &&
bash ted_train.sh &&
bash ted_gen.sh &&
bash web_crawl_pre.sh &&
bash web_crawl_train.sh &&
bash web_crawl_gen.sh &&
bash mixed_pre.sh &&
bash mixed_train.sh &&
bash mixed_gen.sh