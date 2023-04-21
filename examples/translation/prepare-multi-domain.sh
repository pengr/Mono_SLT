#!/bin/bash
# Adapted from https://github.com.cnpmjs.org/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com.cnpmjs.org/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com.cnpmjs.org/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
#DEESCAPE_SPEC_CHAR=$SCRIPTS/tokenizer/deescape-special-chars.perl
#BPEROOT=subword-nmt/subword_nmt
#BPE_TOKENS=40000
URL=https://github.com.cnpmjs.org/pengr/iwslt15.git
GZ=en-de.tgz


URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
    "en-de/train.tags.en-de"
)
DOMAINS=(
    "parliamentary"
    "web_crawl"
    "news"
    "ted"
)


# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ "$1" == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
    OUTDIR=multi_domain
else
    OUTDIR=multi_domain
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
lang_rvs=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
#dev=dev/newstest2013

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

# dowmload wmt14
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi
    if [ ${file: -4} == ".tgz" ]; then
       tar zxvf $file
    elif [ ${file: -4} == ".tar" ]; then
       tar xvf $file
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    for ((i=0;i<${#CORPORA[@]};++i)); do
        f=${CORPORA[i]}
        domain=${DOMAINS[i]}
        rm $tmp/$domain.$l
        # wmt14
        if [ $i -lt 3 ]; then
            cat $orig/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -no-escape -l $l >> $tmp/$domain.tok.$l
        # iwslt15
        else
            cat $orig/$f.$l | \
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
            perl $TOKENIZER -threads 8 -no-escape -l $l > $tmp/$domain.tok.$l
            echo ""
        fi
    done
done

echo "cleaning train data..."
for domain in "${DOMAINS[@]}"; do
    perl $CLEAN -ratio 1.5 $tmp/$domain.tok $src $tgt $tmp/$domain.clean 1 60
done

echo "lower-casing train data..."
for l in $src $tgt; do
    for domain in "${DOMAINS[@]}"; do
        perl $LC < $tmp/$domain.clean.$l > $tmp/$domain.lower.$l
    done
done

echo "splitting wmt14 train, valid and test ..."
for l in $src $tgt; do
    # "parliamentary"
    awk '{if (NR%1001 == 0)  print $0; }' $tmp/${DOMAINS[0]}.lower.$l > $tmp/${DOMAINS[0]}.test.$l
    awk '{if (NR%1000 == 0)  print $0; }' $tmp/${DOMAINS[0]}.lower.$l > $tmp/${DOMAINS[0]}.valid.$l
    awk '{if (NR%1000 != 0 && NR%1001 != 0)  print $0; }' $tmp/${DOMAINS[0]}.lower.$l > $tmp/${DOMAINS[0]}.train.$l
    # "web_crawl"
    awk '{if (NR%1001 == 0)  print $0; }' $tmp/${DOMAINS[1]}.lower.$l > $tmp/${DOMAINS[1]}.test.$l
    awk '{if (NR%1000 == 0)  print $0; }' $tmp/${DOMAINS[1]}.lower.$l > $tmp/${DOMAINS[1]}.valid.$l
    awk '{if (NR%1000 != 0 && NR%1001 != 0)  print $0; }' $tmp/${DOMAINS[1]}.lower.$l > $tmp/${DOMAINS[1]}.train.$l
    # "news"
    awk '{if (NR%101 == 0)  print $0; }' $tmp/${DOMAINS[2]}.lower.$l > $tmp/${DOMAINS[2]}.test.$l
    awk '{if (NR%100 == 0)  print $0; }' $tmp/${DOMAINS[2]}.lower.$l > $tmp/${DOMAINS[2]}.valid.$l
    awk '{if (NR%100 != 0 && NR%101 != 0)  print $0; }' $tmp/${DOMAINS[2]}.lower.$l > $tmp/${DOMAINS[2]}.train.$l
done

echo "pre-processing iwslt15 valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/en-de/IWSLT15.TED*.$l.xml`; do
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

echo "creating iwslt15 train, valid, test..."
for l in $src $tgt; do
    cat $tmp/${DOMAINS[3]}.lower.$l > $tmp/${DOMAINS[3]}.train.$l
    cat $tmp/IWSLT15.TED.tst2012.en-de.$l > $tmp/${DOMAINS[3]}.valid.$l

    cat $tmp/IWSLT15.TED.tst2013.en-de.$l \
        $tmp/IWSLT15.TED.tst2014.en-de.$l \
        > $tmp/${DOMAINS[3]}.test.$l
done

echo "no bpe, moving train, valid, test..."
for L in $src $tgt; do
    for domain in "${DOMAINS[@]}"; do
        for f in train.$L valid.$L test.$L; do
            cp $tmp/$domain.$f $prep/$domain.$f
        done
    done
done