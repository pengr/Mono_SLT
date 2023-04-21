#!/bin/bash
cd ~/mono_slt

# Call scripts/bleu, rouge, meteor scripts to get translation results
for f in valid test; do
    rm -f ../checkpoints/rule_aslg_proc4/pre/$f.txt
    grep ^H ../checkpoints/rule_aslg_proc4/pre/generate-$f.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/rule_aslg_proc4/pre/$f.txt
    if [ $f = test ]; then
        # BLEU-1,2,3,4
        python scripts/bleu.py 1 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
        python scripts/bleu.py 2 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
        python scripts/bleu.py 3 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
        python scripts/bleu.py 4 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
        # ROUGE
        python scripts/rouge.py ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
        # METEOR
        python scripts/meteor.py ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/$f.en
    else
        # BLEU-1,2,3,4
        python scripts/bleu.py 1 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
        python scripts/bleu.py 2 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
        python scripts/bleu.py 3 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
        python scripts/bleu.py 4 ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
        # ROUGE
        python scripts/rouge.py ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
        # METEOR
        python scripts/meteor.py ../checkpoints/rule_aslg_proc4/pre/$f.txt ../data/aslg/dev.en
    fi
done