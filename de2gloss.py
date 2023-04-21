#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.
# Script for transferring English text into Gloss text of sign language

import re, os, argparse, time, html
import stanza
import multiprocessing
from collections import Counter, defaultdict, OrderedDict
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
# nlp = stanza.Pipeline('de', r"../stanza_resources",
#                       tokenize_pretokenized=True, verbose=False,
#                       pos_batch_size=5000, lemma_batch_size=2048, ner_batch_size=2048)

# Customized model with Stanza
# model = torch.load('../stanza_resources/de/lemma/gsd.pt', map_location='cpu')
# _, composite_dict = model['dicts']
#
# # Customize your own dictionary
# external_dict = { ('wolken', 'NOUN'): 'wolke',
#                   ('wolke', 'NOUN'): 'wolke',
#                  ('dritten', 'ADJ'): 'dritte',
#                  ('dritten', 'NOUN'): 'dritte',
#                  ('dritte', 'ADJ'): 'dritte',
#                  ('dritte', 'NOUN'): 'dritte',
#                  ('temperaturen', 'NOUN'): 'temperature',
#                  ('steigenden', 'NOUN'): 'steigen',
#                  ('anfangs', 'ADV'): 'anfang',
#                  ('windig', 'ADJ'): 'wind',
#                  ('windiges', 'ADJ'): 'wind',
#                  ('wind', 'NOUN'): 'wind',
#                  }
# composite_dict.update(external_dict)
# # Save your model and Load your customized model with Stanza
# torch.save(model, '../stanza_resources/de/lemma/gsd_customized.pt')
nlp = stanza.Pipeline('de', r"../stanza_resources",
                      lemma_model_path='../stanza_resources/de/lemma/gsd_customized.pt',
                      tokenize_pretokenized=True, verbose=False,
                      pos_batch_size=5000, lemma_batch_size=2048, ner_batch_size=2048)


# Replace "\s+"->" ", remove the blanks and slice
SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_word(sent):
    sent = SPACE_NORMALIZER.sub(" ", sent)
    return " ".join(sent.strip().split())


# no aggressive hyphen splitting, no-escape and segmentation, doesn't remove bpe
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


# multi-token segmentation
def multi_tokenize(sent):
    sent = sent.split('herb')
    sent = ' herb '.join(sent).split('hoch')
    sent = ' hoch '.join(sent).split('land')
    sent = ' land '.join(sent).split('undzwanzig')
    sent = ' undzwanzig '.join(sent).split('zwanzig')
    sent = ' zwanzig '.join(sent).split('regen')
    sent = ' regen '.join(sent).split('graupel')
    sent = ' graupel '.join(sent).split('mittag')
    sent = ' mittag '.join(sent).split('schnee')
    return ' schnee '.join(sent)


# punctuation and special symbols
def handle_punct_spec(sent):
    # ommit
    # sent = " ".join(sent)
    sent = re.sub('\.$', '', sent)  # 删除末尾的句号
    return sent


def get_stopwords1(args):
    # Load of Spacy and Stanford Stopwords
    with open('scripts/spacy_de_stopwords',encoding='utf-8') as f1, open('scripts/nltk_de_stopwords',encoding='utf-8') as f2:
        spacy_stopwords = [l.strip('\n') for l in f1.readlines()]
        nltk_stopwords = [l.strip('\n') for l in f2.readlines()]
        stopwords = set(spacy_stopwords + nltk_stopwords)

    # 对DE和EN文本小写化,并tokenize
    de_sents = [de_sent.lower().strip().split() for de_sent in open(args.input).readlines()]
    gloss_sents = [gloss_sent.lower().strip().split() for gloss_sent in open(args.reference).readlines()]

    # a.提取在DE文本中出现过的所有停用词, 统计所有单词的词频；
    de_stopwords = []
    de_words = []
    for de_sent in de_sents:
        de_stopword = [w for w in de_sent if w in stopwords]
        de_stopwords += de_stopword
        de_words.extend(de_sent)
    de_wordfreq = Counter(de_words)
    de_stopwords = list(set(de_stopwords))

    # b.统计在Gloss文本中所有单词的词频(做成字典)
    gloss_wordfreq = defaultdict(int)
    for gloss_sent in gloss_sents:
        for w in gloss_sent:
            gloss_wordfreq[w] += 1

    # c. 查看在DE文本中出现过的停用词,在DE文本的词频与Gloss文本的词频差占DE文本词频的百分比;
    # 百分比越高表明,该停用词仅在DE文本出现;超过50%的概率就应该删除
    for de_stopword in de_stopwords:
        diff_freq = de_wordfreq[de_stopword] - gloss_wordfreq[de_stopword]
        diff_pct = diff_freq / de_wordfreq[de_stopword]
        if diff_pct < 0.90:
            de_stopwords.remove(de_stopword)

    return de_stopwords


def get_stopwords2(args):
    assert os.path.exists(args.de_dict) and os.path.exists(args.gloss_dict)  # 检查生成文本和目标字典文本

    # step1. 从德语字典文本(即dict.de.txt,文本格式为:<symbol0> <count0>)加载目标字典类(Dictionary)
    de_stopwords = defaultdict(int)
    with open(args.de_dict, 'r') as f:
        for line in f.readlines():
            word, freq = line.rstrip().rsplit(" ", 1)  # 获得清除空白,以及空格分片的字符列表
            if int(freq) > 0:
                de_stopwords[word] = int(freq)

    # step2. 从Gloss字典文本(即dict.gloss.txt,文本格式为:<symbol0> <count0>)加载目标字典类(Dictionary)
    gloss_stopwords = defaultdict(int)
    with open(args.gloss_dict, 'r') as f:
        for line in f.readlines():
            gloss, freq = line.rstrip().rsplit(" ", 1)  # 获得清除空白,以及空格分片的字符列表
            if int(freq) > 0:
                gloss_stopwords[gloss] = int(freq)

    # step3. 查看在DE文本中出现过的单词,在DE文本的词频与Gloss文本的词频差占DE文本词频的百分比;
    # 百分比越高表明,该停用词仅在DE文本出现;超过50%的概率就应作为停用词
    for de_stopword in list(de_stopwords.keys()):
        diff_freq = de_stopwords[de_stopword] - gloss_stopwords[de_stopword]
        diff_pct = diff_freq / de_stopwords[de_stopword]
        if diff_pct < 0.9:
            del de_stopwords[de_stopword]

    return de_stopwords


# replace specific stopwords
def repl_stopwords(sent):
    # direction word
    sent = sent.replace('nordhälfte', 'nord')
    sent = sent.replace('nordosthälfte', 'nordost').replace('nordöstlich', 'nordost').replace('nordosten', 'nordost')
    sent = sent.replace('südosten', 'suedost').replace('südosthälfte', 'suedost')
    sent = sent.replace('osthälfte', 'ost').replace('osten', 'ost')
    sent = sent.replace('süden', 'sued')
    sent = sent.replace('westen', 'west').replace('westhälfte', 'west')
    sent = sent.replace('norden', 'nord')
    sent = sent.replace('nordwesten', 'nordwest')
    sent = sent.replace('südwesten', 'suedwest')
    # four special letters
    sent = re.sub(r'ä', 'ae', sent)
    sent = re.sub(r'ö', 'oe', sent)
    # sent = re.sub(r'ü', 'ue', sent)
    sent = re.sub(r'ß', 'ss', sent)
    # Synonym words
    sent = sent.replace('wettervorhersage', 'wetter wie-aussehen')
    sent = sent.replace('vor allem', 'besonders')
    sent = sent.replace('kälteste', 'besonders kalt')
    sent = sent.replace('regnet', 'regen')
    sent = sent.replace('erfreuliche', 'freuen')
    sent = sent.replace('und nun', 'jetzt')
    sent = sent.replace('stürmischen', 'sturm')
    sent = sent.replace('sonnenschein', 'sonne').replace('sonnenseite', 'sonne')

    return sent


def remove_stopwords(sents, stopwords):
    new_sents = []
    for sent in sents:
        new_sent = []
        for w in sent.split():
            if w not in stopwords:
                new_sent.append(w)
        new_sents.append(" ".join(new_sent))
    return new_sents


# 把所有字符中的大写字母转换成小写字母
def lower(sent):
    return sent.lower().strip()


# # 对单词进行词性标注(选择universal POS (UPOS) tags,而非treebank-specific POS (XPOS) tags)
# def pos(docs):
#     sents = []
#     sents_pos = []
#     sents_feats = []
#
#     for i, doc in enumerate(docs):
#         for sent in doc.sentences:
#             line = sent.text
#             w_pos= []
#             w_xpos = []
#             w_feats = []
#             for word in sent.words:
#                 w_pos.append(word.pos)
#                 w_xpos.append(word.xpos)
#                 w_feats.append(word.feats)
#
#             sents_pos.append(w_pos)
#             sents_feats.append(w_feats)
#
#     return sents


# # 对单词进行依赖解析
# def depparse(docs):
#     sents = []
#     sents_head = []
#     sents_deprel = []
#
#     for i, doc in enumerate(docs):
#         for sent in doc.sentences:
#             line = sent.text
#             w_head= []
#             w_deprel = []
#
#             for word in sent.words:
#                 w_h = word.head
#                 w_d = word.deprel
#                 w_head.append(w_h)
#                 w_deprel.append(w_d)
#
#             sents_head.append(w_head)
#             sents_deprel.append(w_deprel)
#
#     return sents


# 先首字母大写完成命名实体识别,专有名词识别需首字母大写;
# def ner(docs):
#     sents = []
#     for i, doc in enumerate(docs):
#         for sent in doc.sentences:
#             # for i, sent in enumerate(docs.sentences):
#             # tokens = [token.text[:-2] + 'ium' if token.text.endswith('ia') else \
#             #           token.text[:-2] + 'lum' if token.text.endswith('la') else \
#             #           token.text[:-1] + 'us' if token.text.endswith('i') else \
#             #           token.text for token in sent.tokens if token.ner != 0]
#             tokens = []
#             for token in sent.tokens:
#                 t_ner = token.ner
#                 t_text = token.text
#                 if t_ner != 'O':
#                     if t_text.endswith('ia'):
#                         t_text = t_text[:-2] + 'ium'
#                     elif t_text.endswith('la'):
#                         t_text = t_text[:-2] + 'lum'
#                     elif t_text.endswith('i'):
#                         t_text = t_text[:-1] + 'us'
#                 tokens.append(t_text)
#             sents.append(" ".join(tokens))
#
#     return sents


# 再全部小写进行词形还原;词形还原需全部小写
def lemmatize(docs):
    sents = []
    for i, doc in enumerate(docs):
        for sent in doc.sentences:
            sents.append(" ".join([word.lemma for word in sent.words]))
    return sents


# multiprocessing
# pool = multiprocessing.Pool()
#
# # step1. Tokenize all EN sents
# en_tok = pool.map(tokenize, en_sents)
#
# pool.close()


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

    # Read all DE sents
    with open(args.input, 'r', encoding='utf-8') as f:
        de_sents = f.readlines()
    # Read all Gloss sents
    if args.reference is not None:
        with open(args.reference, 'r', encoding='utf-8') as f:
            gloss_sents = f.readlines()

    # split the whole corpus into multiple shard
    de_sents_shards = split_corpus(de_sents, args.shard_size)
    gloss_sents_shards = split_corpus(gloss_sents, args.shard_size) if args.reference is not None else None

    # reuse three list to save updated sents
    fakegloss_sents = []
    de_sents = []
    gloss_sents = []
    if args.reference:
        for de_sents_shard, gloss_sents_shard in zip(de_sents_shards, gloss_sents_shards):
            # Step1. Attempt reconstructing original data
            # (no aggressive hyphen splitting, no-escape and segmentation, remove bpe)
            if not args.no_reconst_orig:
                fakegloss_sents_shard = list(map(reconst_orig, de_sents_shard))
                print('Reconst_orig Done')

            # step2. Omit and replace punctuation and special symbols
            if not args.no_handle_punct_spec:
                fakegloss_sents_shard = list(map(handle_punct_spec, fakegloss_sents_shard))
                print('Handle_punct_spec Done')

            # Step3. Multi-word Token Expansion　(BPE or Manual)
            if not args.no_multi_tokenize:
                fakegloss_sents_shard = list(map(multi_tokenize, fakegloss_sents_shard))
                print('Multi_tokenize Done')

            # Step4. get stopwords dict, replace stopwords, remove stopwords in DE text
            if not args.no_stopwords:
                # de_stopwords = get_stopwords1(args)
                de_stopwords = get_stopwords2(args)
                fakegloss_sents_shard = list(map(repl_stopwords, fakegloss_sents_shard))
                fakegloss_sents_shard = remove_stopwords(fakegloss_sents_shard, de_stopwords)
                print('Repl_nd_Remove_stopwords Done')

            # Step5. Delete blank lines (Given Reference)
            if not args.no_delete_blank:
                thr_sents = []
                for sent1, sent2, sent3 in zip(de_sents_shard, fakegloss_sents_shard, gloss_sents_shard):
                    if sent2.strip():
                        thr_sents.append((sent1, sent2, sent3))
                de_sents_shard = [sents[0] for sents in thr_sents]
                fakegloss_sents_shard = [sents[1] for sents in thr_sents]
                gloss_sents_shard = [sents[2] for sents in thr_sents]
                print('Delete blank lines Done')

            # step6. Lowercase all letter of each word before lemmatization
            if not args.no_lower:
                fakegloss_sents_shard = list(map(lower, fakegloss_sents_shard))
                print('Lower Done')

            # step7. save the fake gloss, de, gloss shard
            fakegloss_sents += fakegloss_sents_shard
            de_sents += de_sents_shard
            gloss_sents += gloss_sents_shard
    else:
        for de_sents_shard in de_sents_shards:
            # Step1. Attempt reconstructing original data
            # (no aggressive hyphen splitting, no-escape and segmentation, remove bpe)
            if not args.no_reconst_orig:
                fakegloss_sents_shard = list(map(reconst_orig, de_sents_shard))
                print('Reconst_orig Done')

            # step2. Omit and replace punctuation and special symbols
            if not args.no_handle_punct_spec:
                fakegloss_sents_shard = list(map(handle_punct_spec, fakegloss_sents_shard))
                print('Handle_punct_spec Done')

            # Step3. Multi-word Token Expansion　(BPE or Manual)
            if not args.no_multi_tokenize:
                fakegloss_sents_shard = list(map(multi_tokenize, fakegloss_sents_shard))
                print('Multi_tokenize Done')

            # Step4. get stopwords dict, replace stopwords, remove stopwords in DE text
            if not args.no_stopwords:
                # de_stopwords = get_stopwords1(args)
                de_stopwords = get_stopwords2(args)
                fakegloss_sents_shard = list(map(repl_stopwords, fakegloss_sents_shard))
                fakegloss_sents_shard = remove_stopwords(fakegloss_sents_shard, de_stopwords)
                print('Repl_nd_Remove_stopwords Done')

            # Step5. Delete blank lines (Given Reference)
            if not args.no_delete_blank:
                both_sents = []
                for sent1, sent2 in zip(de_sents_shard, fakegloss_sents_shard):
                    if sent2.strip():
                        both_sents.append((sent1, sent2))
                de_sents_shard = [sents[0] for sents in both_sents]
                fakegloss_sents_shard = [sents[1] for sents in both_sents]
                print('Delete blank lines Done')

            # step6. Lowercase all letter of each word before lemmatization
            if not args.no_lower:
                fakegloss_sents_shard = list(map(lower, fakegloss_sents_shard))
                print('Lower Done')

            # step7. save the fake gloss, de, gloss shard
            fakegloss_sents += fakegloss_sents_shard
            de_sents += de_sents_shard

    # Test Gloss tokens matching - Corpus Level
    if args.reference is not None and args.corpus_bleu:
        print(result_corpus(fakegloss_sents, gloss_sents, n=1))
        print(result_corpus(fakegloss_sents, gloss_sents, n=2))
        print(result_corpus(fakegloss_sents, gloss_sents, n=3))
        print(result_corpus(fakegloss_sents, gloss_sents, n=4))

    # Test Gloss tokens matching - Sentence Level
    if args.reference is not None and args.sentence_bleu:
        de = []
        fakegloss = []
        gloss = []
        for id, (l1, l2, l3) in enumerate(zip(de_sents, fakegloss_sents, gloss_sents)):
            sent_bleu = result_sentence(l2.lower(), l3.lower())
            if sent_bleu <= 5:
                # print(f'{sent_bleu}/{id}/{l1}/{l2}')
                de.append(l1)
                fakegloss.append(l2)
                gloss.append(l3)
        print(f'{len(de)} sents')

        # write the lower bleu into output file
        with open(args.output, 'w', encoding='utf-8') as f:
            for de_sent, gloss_sent, fakegloss_sent in zip(de,gloss,fakegloss):
                f.write(de_sent.lower().strip() + '\n')
                f.write(gloss_sent.lower().strip() + '\n')
                f.write(fakegloss_sent.lower().strip() + '\n')

    # Step 8. Write Gloss Sentences to file
    with open(args.input, 'w', encoding='utf-8') as f1, open(args.output, 'w', encoding='utf-8') as f2:
        for de_sent, fakegloss_sent in zip(de_sents,fakegloss_sents):
            f1.write(de_sent.strip() + '\n')
            f2.write(fakegloss_sent.strip() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="de2gloss")
    parser.add_argument("-input", metavar="SRC", required=True,
                        help="Path of the source file (German text)")
    parser.add_argument("-output", metavar="TGT", required=True,
                        help="Path of the target file (Transferred Gloss text)")
    parser.add_argument("-reference", metavar="REF", default=None,
                        help="Path of the reference file (Reference Gloss text)")
    parser.add_argument("-de_dict", metavar="REF", default=None,
                        help="Path of the german dict file")
    parser.add_argument("-gloss_dict", metavar="REF", default=None,
                        help="Path of the gloss dict file")
    parser.add_argument('-shard_size', type=int, metavar='D', default=1000000,
                        help="Divide corpus into smaller multiple corpus files,"
                             "shard_size>0 means segment dataset into multiple shards")
    parser.add_argument('-no_reconst_orig', default=False, action="store_true",
                        help='Attempt reconstructing original data'
                             '(no aggressive hyphen splitting, no-escape and segmentation, remove bpe)')
    parser.add_argument('-no_tokenize', default=False, action="store_true", help='Tokenize all DE sents')
    parser.add_argument('-no_handle_punct_spec', default=False, action="store_true",
                        help='Omit and replace punctuation and special symbols')
    parser.add_argument('-no_multi_tokenize', default=False, action="store_true", help='Multi_Tokenize all DE sents')
    parser.add_argument('-no_stopwords', default=False, action="store_true", help='Get stopwords')
    parser.add_argument('-no_delete_blank', default=False, action="store_true", help='Delete all blank lines')
    parser.add_argument('-no_lower', default=False, action="store_true", help='lower')
    # Stanza Option (unused now)
    # parser.add_argument('-no_lemmatize', default=False, action="store_true",
    #                     help='Lemmatization')
    # parser.add_argument('-no_pos', default=False, action="store_true", help='part-of-speeching')
    # parser.add_argument('-batch_size', type=float, metavar='D', default=1024,
    #                     help='Batch size for text processing in stanza (sentences/per batch)')
    parser.add_argument('-corpus_bleu', default=False, action="store_true", help='Corpus-level bleu')
    parser.add_argument('-sentence_bleu', default=False, action="store_true", help='Sentence-level bleu')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print('Cost {}s'.format(time.time()-start_time))