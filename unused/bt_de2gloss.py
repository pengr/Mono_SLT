#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.
# Script for transferring English text into Gloss text of sign language

import re, os, argparse, time, html
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
from itertools import groupby


# no aggressive hyphen splitting, no-escape and segmentation, doesn't remove bpe
mapper = {'"': '``'}
def tokenize(sent):
    sent = sent.replace("@-@", "-")
    tokens = html.unescape(sent.strip()).split()
    tokens = list(map(lambda t: mapper.get(t, t), tokens))
    return tokens


# punctuation and special symbols
def remove_spec(sent):
    # ommit
    sent = " ".join(sent)
    sent = re.sub('\.$', '', sent)  # 删除末尾的句号
    # sent = re.sub('\\_\_off\_\_$', '', sent)  # 删除末尾的句号
    sent = re.sub('\_\_[a-zA-Z0-9?]*\_\_', '', sent)  # 删除所有__xx__的单词
    sent = sent.replace('<unk>', '') # 删除<unk>
    sent = " ".join([x[0] for x in groupby(sent.split())]) # 去除连续重复出现的单词
    return sent


# 把所有字符中的大写字母转换成小写字母
def lower(sent):
    return sent.lower().strip()

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


def main(args):
    assert os.path.exists(args.input_de) and os.path.exists(args.input_gloss) # check the path

    # Read all Fake Gloss sents
    with open(args.input_de, 'r', encoding='utf-8') as f1, open(args.input_gloss, 'r', encoding='utf-8') as f2:
        de_sents = f1.readlines()
        fakegloss_sents = f2.readlines()
    # Read all Gloss sents
    if args.reference is not None:
        with open(args.reference, 'r', encoding='utf-8') as f:
            gloss_sents = f.readlines()

    # step1. Tokenize all DE sents (no aggressive hyphen splitting, no-escape and segmentation)
    if not args.no_tokenize:
        fakegloss_sents = list(map(tokenize, fakegloss_sents))
        print('Tokenize Done')

    # step2. remove special symbols
    if not args.no_remove_spec:
        fakegloss_sents = list(map(remove_spec, fakegloss_sents))
        print('Remove_spec Done')

    # step3. Lowercase all letter of each word before lemmatization
    if not args.no_lower:
        fakegloss_sents = list(map(lower, fakegloss_sents))
        print('Lower Done')

    # Step4. Delete blank lines (Given Reference)
    if args.reference is not None:
        thr_sents = []
        for sent1, sent2, sent3 in zip(de_sents, fakegloss_sents, gloss_sents):
            if sent2.strip():
                thr_sents.append((sent1, sent2, sent3))
        de_sents = [sents[0] for sents in thr_sents]
        fakegloss_sents = [sents[1] for sents in thr_sents]
        gloss_sents = [sents[2] for sents in thr_sents]
    else:
        both_sents = []
        for sent1, sent2 in zip(de_sents, fakegloss_sents):
            if sent2.strip():
                both_sents.append((sent1, sent2))
        de_sents = [sents[0] for sents in both_sents]
        fakegloss_sents = [sents[1] for sents in both_sents]
    print('Delete blank lines Done')

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
            for de_sent, gloss_sent, fakegloss_sent in zip(de, gloss, fakegloss):
                f.write(de_sent.lower().strip() + '\n')
                f.write(gloss_sent.lower().strip() + '\n')
                f.write(fakegloss_sent.lower().strip() + '\n')

    # Step 5. Write Gloss Sentences to file
    with open(args.input_de, 'w', encoding='utf-8') as f1, open(args.input_gloss, 'w', encoding='utf-8') as f2:
        for de_sent, fakegloss_sent in zip(de_sents, fakegloss_sents):
            f1.write(de_sent.strip() + '\n')
            f2.write(fakegloss_sent.strip() + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="de2gloss")
    parser.add_argument("-input_de", metavar="SRC", required=True,
                        help="Path of the source file (German text)")
    parser.add_argument("-input_gloss", metavar="SRC", required=True,
                        help="Path of the source file (bt Gloss text)")
    parser.add_argument("-reference", metavar="REF", default=None,
                        help="Path of the reference file (Reference Gloss text)")
    parser.add_argument('-no_tokenize', default=False, action="store_true", help='Tokenize all DE sents')
    parser.add_argument('-no_remove_spec', default=False, action="store_true", help='Remove special words')
    parser.add_argument('--batch_size', type=float, metavar='D', default=1024,
                        help='Batch size for text processing in stanza (sentences/per batch)')
    parser.add_argument('-no_lower', default=False, action="store_true", help='lower')
    parser.add_argument('-corpus_bleu', default=False, action="store_true", help='Corpus-level bleu')
    parser.add_argument('-sentence_bleu', default=False, action="store_true", help='Sentence-level bleu')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print('Cost {}s'.format(time.time()-start_time))