#!/bin/bash
start=$(date +%s)
sh bt_pre.sh &&
sh bt_train.sh &&
sh bt_gen.sh
end=$(date +%s)
duration=$(( end - start ))
echo cost ${duration} seconds. >> bt_phoenix_time.txt