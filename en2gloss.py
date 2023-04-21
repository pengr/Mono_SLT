#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.
# Script for transferring English text into Gloss text of sign language

import re, os, argparse, time, html
import stanza
import multiprocessing
from collections import Counter, defaultdict
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
from typing import List
from stanza.models.common.doc import Document
import toma
import torch
from stanza_batch import batch
from itertools import islice
import gc


# toma requires the first argument of the method to be the batch size
def run_batch(batch_size: int, stanza_nlp: stanza.Pipeline, data: List[str]
              ) -> List[Document]:
    # So that we can see what the batch size changes to.
    print(batch_size)
    return [doc for doc in batch(data, stanza_nlp, batch_size=batch_size)]


# Default model
# stanza.download('en') # download English model
# nlp = stanza.Pipeline('en', r"../stanza_resources",
#                       tokenize_pretokenized=True,  verbose=False)

# # Customized model with Stanza
# model = torch.load('../stanza_resources/en/lemma/combined.pt', map_location='cpu')
# _, composite_dict = model['dicts']
#
# # Customize your own dictionary
# external_dict = {('these', 'DET'): 'these',
#                  ('these', 'PRON'): 'these',
#                  ('those', 'DET'): 'those',
#                  ('those', 'PRON'): 'those',
#                  ('prevailing', 'ADJ'): 'prevail',
#                  ('voting', 'NOUN'): 'vote',
#                  ('negotiating', 'NOUN'): 'negotiate',
#                  ('sitting', 'NOUN'): 'sit',
#                  ('warming', 'NOUN'): 'warm',
#                  ('warning', 'NOUN'): 'warn',
#                  ('monitoring', 'NOUN'): 'monitor',
#                  ('labelling', 'NOUN'): 'label',
#                  ('living', 'NOUN'): 'live',
#                  ('driving', 'NOUN'): 'drive',
#                  ('opening', 'NOUN'): 'open',
#                  ('writing', 'NOUN'): 'write',
#                  ('suffering', 'NOUN'): 'suffer',
#                  ('mainstreaming', 'NOUN'): 'mainstream',
#                  ('working', 'NOUN'): 'work',
#                  ('gathering', 'NOUN'): 'gathering',
#                  }
# composite_dict.update(external_dict)
# # Save your model and Load your customized model with Stanza
# torch.save(model, '../stanza_resources/en/lemma/combined_customized.pt')
nlp = stanza.Pipeline('en', r"../stanza_resources",
                      lemma_model_path='../stanza_resources/en/lemma/combined_customized.pt',
                      processors='tokenize, pos, lemma, ner',
                      tokenize_pretokenized=True, verbose=False,
                      pos_batch_size=5000, lemma_batch_size=2048, ner_batch_size=2048)


# Replace "\s+"->" ", remove the blanks and slice
SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_word(sent):
    sent = SPACE_NORMALIZER.sub(" ", sent)
    return " ".join(sent.strip().split())


# no aggressive hyphen splitting, no-escape and segmentation
mapper = {'"': '``'}
def tokenize(sent):
    sent = sent.replace("@-@", "-")
    tokens = html.unescape(sent.strip()).split()
    tokens = list(map(lambda t: mapper.get(t, t), tokens))
    return tokens


def reconst_orig(sent, separator='@@'):
    tokens = tokenize(sent)
    word, words = [], []
    for tok in tokens:
        if tok.endswith(separator):
            tok = tok.strip(separator)
            word.append(tok)
        else:
            word.append(tok)
            words.append(''.join(word))
            word = []
    sentence = ' '.join(words)
    return sentence


# Extract the last clause
def extract_clause(sent):
    sent = sent.split(" . ")[-1]
    return sent


def abbrev_repl(sent):
    sent = sent.replace('cannot', 'can not')
    sent = re.sub(r'won\'t', 'will not', sent)
    sent = re.sub(r'can\'t', 'can not', sent)
    sent = re.sub(r'i\'m', 'i am', sent)
    sent = re.sub(r'ain\'t', 'is not', sent)
    sent = re.sub(r'(\w+)\'ll', '\g<1> will', sent)
    sent = re.sub(r'(\w+)n\'t', '\g<1> not', sent)
    sent = re.sub(r'(\w+)\'ve', '\g<1> have', sent)
    sent = re.sub(r'(\w+)\'s', '\g<1> poss', sent)
    sent = re.sub(r'(\w+)\'re', '\g<1> are', sent)
    sent = re.sub(r'(\w+)\'d', '\g<1> would', sent)
    return sent


# punctuation and special symbols
# 不变:!"#$%&\'()?+,.:;<=>@\^_|—~￥//‖～§→{}★ 和中文符号：，。、…【】《》？“”‘’！（）和其他语言à转为à`
# 省略:- /
# 替换:*转\*  [转lrb- ]转rrb- \s+转" "
def handle_punct_spec(sent):
    # ommit
    sent = re.sub(r'[\-\/]\s+', '', sent)  # \s+清除ommit后留下的空格
    # replace
    sent = re.sub(r'\.\.', '.', sent)
    sent = re.sub(r'\*', '\\*', sent)
    sent = re.sub(r'\[', 'lrb-', sent)
    sent = re.sub(r'\]', 'rrb-', sent)
    return sent


# 把每个单词的第一个字母转化为大写,其余小写
def title(sent):
    return sent.title().strip()


# 把所有字符中的大写字母转换成小写字母
def lower(sent):
    return sent.lower().strip()


# 先首字母大写完成命名实体识别,专有名词识别需首字母大写;
def ner(docs):
    sents = []
    for i, doc in enumerate(docs):
        for sent in doc.sentences:
            # for i, sent in enumerate(docs.sentences):
            # tokens = [token.text[:-2] + 'ium' if token.text.endswith('ia') else \
            #           token.text[:-2] + 'lum' if token.text.endswith('la') else \
            #           token.text[:-1] + 'us' if token.text.endswith('i') else \
            #           token.text for token in sent.tokens if token.ner != 0]
            tokens = []
            for token in sent.tokens:
                t_ner = token.ner
                t_text = token.text
                if t_ner != 'O':
                    if t_text.endswith('ia'):
                        t_text = t_text[:-2] + 'ium'
                    elif t_text.endswith('la'):
                        t_text = t_text[:-2] + 'lum'
                    elif t_text.endswith('i'):
                        t_text = t_text[:-1] + 'us'
                tokens.append(t_text)
            sents.append(" ".join(tokens))

    return sents


# 再全部小写进行词形还原;词形还原需全部小写
def lemmatize(docs):
    sents = []
    for i, doc in enumerate(docs):
        for sent in doc.sentences:
            sents.append(" ".join([word.lemma for word in sent.words]))
    return sents


# function words
def ommit_func(sent):
    sent = re.sub(r'the', '', sent)  # 删除出现'the'的字符
    sent = re.sub(r'\s+', ' ', sent)  # \s+清除ommit后留下的空格,不能写成'the\s+',这样无法删除'they'

    # 固定虚词作为单词整体,需在字符串首/中间/末尾分别替换,且不能留下不必要的空格
    sent = re.sub('^a\s+|\s+a$', '', sent)  # 删除'a'单词
    sent = re.sub(r'\s+a\s+', ' ', sent)  # 删除'a'单词
    sent = re.sub('^an\s+|\s+an$', '', sent)  # 删除'an'单词
    sent = re.sub(r'\s+an\s+', ' ', sent)  # 删除'an'单词
    sent = re.sub('^of\s+|\s+of$', '', sent)  # 删除'of'单词
    sent = re.sub(r'\s+of\s+', ' ', sent)  # 删除'of'单词
    return sent


# multiprocessing
# pool = multiprocessing.Pool()
#
# # step1. Tokenize all EN sents
# en_tok = pool.map(tokenize, en_sents)
#
# pool.close()


def get_stopwords(en_sents, gloss_sents):
    # Load of NLTK and Stanford Stopwords
    with open('scripts/stanford_en_stopwords') as f1, open('scripts/nltk_en_stopwords') as f2:
        stanford_stopwords = [l.strip('\n') for l in f1.readlines()]
        nltk_stopwords = [l.strip('\n') for l in f2.readlines()]
        stopwords = set(stanford_stopwords + nltk_stopwords)

    # 对DE和EN文本小写化,并tokenize
    en_sents = [en_sent.lower().strip().split() for en_sent in en_sents]
    gloss_sents = [gloss_sent.lower().strip().split() for gloss_sent in gloss_sents]

    # a.提取EN文本中所有停用词, 统计词频；
    en_stopwords = []
    en_words = []
    for en_sent in en_sents.lower():
        en_stopword = [w for w in en_sent if w in stopwords]
        en_stopwords += en_stopword
        en_words.extend(en_sent)
    en_wordfreq = Counter(en_words)
    en_stopwords = set(en_stopwords)

    # b.统计Gloss中所有单词的词频(做成字典)
    gloss_wordfreq = defaultdict(int)
    for gloss_sent in gloss_sents.lower():
        for w in gloss_sent:
            gloss_wordfreq[w] += 1

    # c. 查看EN文本停用词在Gloss的词频
    with open('scripts/en_stopwords', 'w', encoding='utf-8') as f:
        for en_stopword in en_stopwords:
            diff_freq = en_wordfreq[en_stopword] - gloss_wordfreq[en_stopword]
            diff_pct = diff_freq / en_wordfreq[en_stopword]
            if diff_pct >= 0.90:
                info = '{} {} {} '.format(en_stopword, en_wordfreq[en_stopword],
                                          gloss_wordfreq[en_stopword]) + "{:.2f}".format(diff_pct * 100)
                f.write(info + '\n')


# 整个语料库的bleu分
def result_corpus(sys, ref, n=4):
    weights = [1 / n] * n + [0] * (4 - n)
    pred = [p.lower().split() for p in sys]
    target = [[t.lower().split()] for t in ref]
    return corpus_bleu(target, pred, weights=weights) * 100


# 每个句子的bleu分
def result_sentence(sys, ref, order=4):
    if order != 4:
        raise NotImplementedError
    return sacrebleu.sentence_bleu(sys, [ref]).score


# 切分语料库为多个shard
def split_corpus(sents, shard_size):
    if shard_size <= 0:  # 若分片大小为负,则默认不分片
        yield sents
    else:
        sents = iter(sents)
        while True:      # 通过循环一次性分好数据集, 并返回分好的每一片数据集
            shard = list(islice(sents, shard_size))
            if not shard:
                return
            yield shard


def main(args):
    assert os.path.exists(args.input)  # check the path

    # Read all EN sents
    with open(args.input, 'r', encoding='utf-8') as f:
        en_sents = f.readlines()
    # Read all Gloss sents
    if args.reference is not None:
        with open(args.reference, 'r', encoding='utf-8') as f:
            gloss_sents = f.readlines()

    # get stopwords case of text
    if args.get_stopwords:
        get_stopwords(en_sents, gloss_sents)

    # split the whole corpus into multiple shard
    fakegloss_sents = []
    en_sents_shards = split_corpus(en_sents, args.shard_size)
    for en_sents_shard in en_sents_shards:
        # Step1. Attempt reconstructing original data
        # (no aggressive hyphen splitting, no-escape and segmentation, remove bpe)
        if not args.no_reconst_orig:
            fakegloss_sents_shard = list(map(reconst_orig, en_sents_shard))
            print('Reconst_orig Done')

        # step2. extract the last clause
        if not args.no_extract_clause:
            fakegloss_sents_shard = list(map(extract_clause, fakegloss_sents_shard))
            print('Extract_clause Done')

        # step3. Abbreviation replacement
        if not args.no_abbrev_repl:
            fakegloss_sents_shard = list(map(abbrev_repl, fakegloss_sents_shard))
            print('Abbrev_repl Done')

        # step4. Omit and replace punctuation and special symbols
        if not args.no_handle_punct_spec:
            fakegloss_sents_shard = list(map(handle_punct_spec, fakegloss_sents_shard))
            print('Handle_punct_spec Done')

        # step5. Capitalized first letter of each word before the NER
        if not args.no_title:
            fakegloss_sents_shard = list(map(title, fakegloss_sents_shard))
            print('Title Done')

        # step6. Name Entity Recognition
        # Create the document batch
        if not args.no_ner:
            docs: List[Document] = toma.simple.batch(run_batch, args.batch_size, nlp, fakegloss_sents_shard)
            # docs = nlp("\n\n".join(fakegloss_sents))  # Create the document batch
            fakegloss_sents_shard = ner(docs)
            print('Ner Done')
            # 删除docs,腾出内存来
            del docs
            gc.collect()

        # step7. Lowercase all letter of each word before lemmatization
        if not args.no_lower:
            fakegloss_sents_shard = list(map(lower, fakegloss_sents_shard))
            print('Lower Done')

        # step8. Lemmatization
        if not args.no_lemmatize:
            docs: List[Document] = toma.simple.batch(run_batch, args.batch_size, nlp, fakegloss_sents_shard)
            # docs = nlp("\n\n".join(fakegloss_sents))  # Create the document batch
            fakegloss_sents_shard = lemmatize(docs)
            print('Lemmatize Done')
            # 删除docs,腾出内存来
            del docs
            gc.collect()

        # step9. Omit function words
        if not args.no_ommit_func:
            fakegloss_sents_shard = list(map(ommit_func, fakegloss_sents_shard))
            print('Ommit_func Done')

        # step10. save the fake gloss shard
        fakegloss_sents += fakegloss_sents_shard

    # Test Gloss tokens matching - Corpus Level
    if args.reference is not None and args.corpus_bleu:
        print(result_corpus(fakegloss_sents, gloss_sents, n=1))
        print(result_corpus(fakegloss_sents, gloss_sents, n=2))
        print(result_corpus(fakegloss_sents, gloss_sents, n=3))
        print(result_corpus(fakegloss_sents, gloss_sents, n=4))

    # Test Gloss tokens matching - Sentence Level
    if args.reference is not None and args.sentence_bleu:
        en = []
        fakegloss = []
        gloss = []
        for id, (l1, l2, l3) in enumerate(zip(en_sents, fakegloss_sents, gloss_sents)):
            sent_bleu = result_sentence(l2.lower(), l3.lower())
            if sent_bleu <= 90:
                # print(f'{sent_bleu}/{id}/{l1}/{l2}')
                en.append(l1)
                fakegloss.append(l2)
                gloss.append(l3)
        print(f'{len(en)} sents')

    # Step 11. Write Gloss Sentences to file
    with open(args.output, 'w', encoding='utf-8') as f:
        for fakegloss_sent in fakegloss_sents:
            f.write(fakegloss_sent.strip() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="en2gloss")
    parser.add_argument("-input", metavar="SRC", required=True,
                        help="Path of the source file (English text)")
    parser.add_argument("-output", metavar="TGT", required=True,
                        help="Path of the target file (Transferred Gloss text)")
    parser.add_argument("-reference", metavar="REF", default=None,
                        help="Path of the reference file (Reference Gloss text)")
    parser.add_argument('-shard_size', type=int, metavar='D', default=1000000,
                        help="Divide corpus into smaller multiple corpus files,"
                        "shard_size>0 means segment dataset into multiple shards")
    parser.add_argument('-no_reconst_orig', default=False, action="store_true",
                        help='Attempt reconstructing original data'
                             '(no aggressive hyphen splitting, no-escape and segmentation, remove bpe)')
    parser.add_argument('-no_tokenize', default=False, action="store_true", help='Tokenize all EN sents')
    parser.add_argument('-no_extract_clause', default=False, action="store_true", help='extract the last clause')
    parser.add_argument('-no_abbrev_repl', default=False, action="store_true", help='Abbreviation replacement')
    parser.add_argument('-no_handle_punct_spec', default=False, action="store_true",
                        help='Omit and replace punctuation and special symbols')
    parser.add_argument('-no_title', default=False, action="store_true",
                        help='Capitalized first letter of each word before the NER')
    parser.add_argument('-batch_size', type=int, metavar='D', default=1024,
                        help='Batch size for text processing in stanza (sentences/per batch)')
    parser.add_argument('-no_ner', default=False, action="store_true",
                        help='Name Entity Recognition')
    parser.add_argument('-no_lower', default=False, action="store_true", help='lower')
    parser.add_argument('-no_lemmatize', default=False, action="store_true",
                        help='Lemmatization')
    parser.add_argument('-no_ommit_func', default=False, action="store_true", help='Omit function words')
    parser.add_argument('-corpus_bleu', default=False, action="store_true", help='Corpus-level bleu')
    parser.add_argument('-sentence_bleu', default=False, action="store_true", help='Sentence-level bleu')
    parser.add_argument('-get_stopwords', default=False, action="store_true", help='Get stopwords')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print('Cost {}s'.format(time.time() - start_time))