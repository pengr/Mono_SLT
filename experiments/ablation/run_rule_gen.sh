#!/bin/bash
cd ~/mono_slt
PHOENIX_DIR=../data/phoenix2014T
WMT14_DIR=../data/wmt14_en_de

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule1'
mkdir -p $WMT14_DIR/rule1
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule1/train.gloss_proc -no_extract_clause
echo 'no_extract_clause done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule2'
mkdir -p $WMT14_DIR/rule2
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule2/train.gloss_proc -no_abbrev_repl
echo 'no_abbrev_repl done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule3'
mkdir -p $WMT14_DIR/rule3
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule3/train.gloss_proc -no_handle_punct_spec
echo 'no_handle_punct_spec done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule4'
mkdir -p $WMT14_DIR/rule4
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule4/train.gloss_proc -no_title -no_ner
echo 'no_title no_ner done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule5'
mkdir -p $WMT14_DIR/rule5
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule5/train.gloss_proc -no_lemmatize
echo 'no_lemmatize done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
echo 'en2gloss rule6'
mkdir -p $WMT14_DIR/rule6
python en2gloss.py -input $WMT14_DIR/train.en -output $WMT14_DIR/rule6/train.gloss_proc -no_ommit_func
echo 'no_ommit_func done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
mkdir -p $WMT14_DIR/rule1
cp -f $WMT14_DIR/train.de $WMT14_DIR/rule1/train.de
echo 'de2gloss rule1'
python de2gloss.py -input $WMT14_DIR/rule1/train.de -output  $WMT14_DIR/rule1/train.gloss \
-de_dict $PHOENIX_DIR/phoenix/dict.de.txt -gloss_dict $PHOENIX_DIR/phoenix/dict.gloss.txt -no_handle_punct_spec
echo 'no_handle_punct_spec done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
mkdir -p $WMT14_DIR/rule2
cp -f $WMT14_DIR/train.de $WMT14_DIR/rule2/train.de
echo 'de2gloss rule2'
python de2gloss.py -input $WMT14_DIR/rule2/train.de -output  $WMT14_DIR/rule2/train.gloss \
-de_dict $PHOENIX_DIR/phoenix/dict.de.txt -gloss_dict $PHOENIX_DIR/phoenix/dict.gloss.txt -no_multi_tokenize
echo 'no_multi_tokenize done'

# Generate the wmt14 Gloss data by Transformation Rule (Train)
mkdir -p $WMT14_DIR/rule3
cp -f $WMT14_DIR/train.de $WMT14_DIR/rule3/train.de
echo 'de2gloss rule3'
python de2gloss.py -input $WMT14_DIR/rule3/train.de -output  $WMT14_DIR/rule3/train.gloss \
-de_dict $PHOENIX_DIR/phoenix/dict.de.txt -gloss_dict $PHOENIX_DIR/phoenix/dict.gloss.txt -no_stopwords
echo 'no_stopwords done'