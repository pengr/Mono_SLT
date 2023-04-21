#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000
URL=https://github.com.cnpmjs.org/pengr/iwslt15.git
GZ=en-de.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=iwslt15.tokenized.en-de
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig
# download iwslt15
if [ -f $GZ ]; then
    echo "$GZ already exists, skipping download"
else
    echo "Downloading data from ${URL}..."
    git clone $URL
    mv -f iwslt15/$GZ $GZ
    rm -rf iwslt15
    if [ -f $GZ ]; then
        echo "Data successfully downloaded."
    else
        echo "Data not successfully downloaded."
        exit
    fi
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<keywords>' | \
    grep -v '<speaker>' | \
    grep -v '<talkid>' | \
    grep -v '<reviewer>' | \
    grep -v '</reviewer>' | \
    grep -v '<translator>' | \
    grep -v '</translator>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -no-escape -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 60
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT15.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -no-escape -l $l | \
    perl $LC > $f
    echo ""
    done
done

echo "creating train, valid, test..."
for l in $src $tgt; do
    cat $tmp/train.tags.en-de.$l > $tmp/train.$l
    cat $tmp/IWSLT15.TED.tst2012.en-de.$l > $tmp/valid.$l

    cat $tmp/IWSLT15.TED.tst2013.en-de.$l \
        $tmp/IWSLT15.TED.tst2014.en-de.$l \
        > $tmp/test.$l
done

#TRAIN=$tmp/train.en-de
#BPE_CODE=$prep/code
#rm -f $TRAIN
#for l in $src $tgt; do
#    cat $tmp/train.$l >> $TRAIN
#done

#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

#for L in $src $tgt; do
#    for f in train.$L valid.$L test.$L; do
#        echo "apply_bpe.py to ${f}..."
#        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
#    done
#done

echo "no bpe, moving train, valid, test..."
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        cp $tmp/$f  $prep/$f
    done
done