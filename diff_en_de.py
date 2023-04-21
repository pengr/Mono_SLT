# -*- coding: utf-8 -*-
from collections import Counter
import re, logging, argparse


SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def diff_en_de(args):
    # initialize two counter
    text_counter = Counter()
    gloss_counter = Counter()

    with open(file=args.text, encoding='utf-8') as f1, open(file=args.gloss, encoding='utf-8') as f2:
        for t_line, g_line in zip(f1.readlines(), f2.readlines()):

            # count the number of word in text seq and gloss seq
            t_counts = Counter(t_line)
            g_counts = Counter(g_line)

            for g, g_count in g_counts.items():
                # get the occurrences of current gloss in text seq
                t_count = t_counts[g]
                # Get the corrected translation num of current gloss in text seq, < the occurrences of itself in gloss seq
                gloss_count = min(t_count, g_count)
                # update the occurrences into two counter
                gloss_counter.update({g: gloss_count})
                text_counter.update({g: g_count})

    word_acc = sum(gloss_counter.values()) / sum(text_counter.values())
    print('word acc: %.2f %%' %(word_acc*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diff_en_de")
    parser.add_argument("--gloss", metavar="SRC", required=True,
                        help="Path of the gloss file")
    parser.add_argument("--text", metavar="TGT", required=True,
                        help="Path of the text file")
    args = parser.parse_args()

    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    diff_en_de(args)