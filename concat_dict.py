#!/usr/bin/env python
from collections import OrderedDict
import sys

def main(argv):
    with open(argv[0], 'r', encoding='utf8') as dict_f1, \
         open(argv[1], 'r', encoding='utf8') as dict_f2:

        # 创建输入字典1
        dict1 = OrderedDict()
        for line1 in dict_f1.readlines():
            word, field = line1.rstrip().rsplit(" ", 1)  # 获得清除空白,以及空格分片的字符列表
            count = int(field)  # 获得当前单词的字频
            dict1.update({word:count})

        # 创建输入字典2
        dict2 = OrderedDict()
        for line2 in dict_f2.readlines():
            word, field = line2.rstrip().rsplit(" ", 1)  # 获得清除空白,以及空格分片的字符列表
            count = int(field)  # 获得当前单词的字频
            dict2.update({word: count})

        # 遍历字典2的键值对,查找未在字典1出现过的键,将该键值对更新到字典1内
        for word, count in dict2.items():
            if word not in dict1.keys():
                dict1.update({word: count})

    with open(argv[0], 'w', encoding='utf8') as dict_f1:
        # 将更新后的字典1写入新字典文件内
        for word, conut in dict1.items():
            dict_f1.write(f"{word} {conut}\n")

if "__main__" == __name__:
    main(sys.argv[1:])