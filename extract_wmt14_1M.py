import random
import numpy as np
import sys


def main(argv):
    max_len = 60

    with open(argv[0], 'r', encoding='utf8') as f1, \
         open(argv[1], 'r', encoding='utf8') as f2, \
         open(argv[2], 'w', encoding='utf8') as f3, \
         open(argv[3], 'w', encoding='utf8') as f4:

        en_ls = []
        de_ls = []
        for en_l, de_l in zip(f1.readlines(), f2.readlines()):
            en_l = en_l.strip().split()
            de_l = de_l.strip().split()

            if (len(en_l) <= max_len) and (len(de_l) <= max_len):
                en_ls.append(en_l)
                de_ls.append(de_l)

        ids = np.random.choice(np.arange(len(en_ls)), 1000000, replace=False)

        choice_en_ls = [en_ls[id] for id in ids]
        choice_de_ls = [de_ls[id] for id in ids]

        for en_l, de_l in zip(choice_en_ls, choice_de_ls):
            f3.write(" ".join(en_l) + '\n')
            f4.write(" ".join(de_l) + '\n')


if "__main__" == __name__:
    main(sys.argv[1:])