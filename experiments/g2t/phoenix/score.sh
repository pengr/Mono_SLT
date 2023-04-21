#!/bin/bash
cd ~/mono_slt

# Call scripts/bleu, rouge, meteor scripts to get translation results
for f in valid test; do
    rm -f ../checkpoints/phoenix/$f.txt
    grep ^H ../checkpoints/phoenix/generate-$f.txt | sort -n -k 2 -t '-' | cut -f 3 >> ../checkpoints/phoenix/$f.txt
    if [ $f = test ]; then
        # BLEU-1,2,3,4
        python scripts/bleu.py 1 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
        python scripts/bleu.py 2 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
        python scripts/bleu.py 3 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
        python scripts/bleu.py 4 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
        # ROUGE
        python scripts/rouge.py ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
        # METEOR
        python scripts/meteor.py ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/$f.de
    else
        # BLEU-1,2,3,4
        python scripts/bleu.py 1 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
        python scripts/bleu.py 2 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
        python scripts/bleu.py 3 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
        python scripts/bleu.py 4 ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
        # ROUGE
        python scripts/rouge.py ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
        # METEOR
        python scripts/meteor.py ../checkpoints/phoenix/$f.txt ../data/phoenix2014T/dev.de
    fi
done