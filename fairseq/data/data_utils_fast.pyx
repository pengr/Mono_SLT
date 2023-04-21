# cython: language_level=3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


cdef _is_batch_full(list batch, long num_tokens, long max_tokens, long max_sentences):
    if len(batch) == 0:  # 若当前的batch为空,则返回0
        return 0
    if max_sentences > 0 and len(batch) == max_sentences:  # 若一个批次的最大句子数不为空(采用sents-level batch),且当前batch的示例数等于最大句子数,则返回1
        return 1
    if max_tokens > 0 and num_tokens > max_tokens:  # 若一个批次的最大tokens数不为空(采用tokens-level batch),且当前batch的示例数等于最大tokens数,则返回1
        return 1
    return 0  # 若上述三种均不满足,则返回0


@cython.cdivision(True)
cpdef list batch_by_size_fast(
    np.ndarray[DTYPE_t, ndim=1] indices,
    num_tokens_fn,
    long max_tokens,
    long max_sentences,
    int bsz_mult,
):
    cdef long sample_len = 0     # 初始化该批次所有示例的最大长度为0
    cdef list sample_lens = []   # 记录所有句子长度的列表
    cdef list batch = []         # 用于存储该批次的所有句子示例的编号
    cdef list batches = []      # 用于存储所有批次的列表,每个批次中有句子的编号
    cdef long mod_len
    cdef long i
    cdef long idx
    cdef long num_tokens
    cdef DTYPE_t[:] indices_view = indices

    for i in range(len(indices_view)):  # 遍历非过长句子的索引有序列表的ndarray一维数组
        idx = indices_view[i]           # 取出当前所遍历句子示例的编号
        num_tokens = num_tokens_fn(idx)  # 返回当前所遍历句子示例的token数(源和目标之间的最大值)
        sample_lens.append(num_tokens)   # 将当前示例的长度添加进所有句子长度的列表
        sample_len = max(sample_len, num_tokens)  # 不断比较,得到当前批次所有示例的最大长度

        assert max_tokens <= 0 or sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )  # 若当前batch中所有示例的最大长度超过最大tokens数,则报错
        num_tokens = (len(batch) + 1) * sample_len  # 由当前batch中所有句子的总计token数来更新num_tokens

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):  # 用于判断当前batch是否已满,满了为True(tokens->是否达到max_tokens,sents->是否达到max_sentences)
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )   # 获得当前batch的示例数(tokens数)除以N的余数,若当前batch中所有句子的总计token数超过max_tokens时,存在余数
            # 默认情况mod_len为len(batch)
            batches.append(batch[:mod_len])  # 将当前batch添加到用于存储所有批次的列表中;
            batch = batch[mod_len:]  # 提取剩余句子的tokens到batch;
            sample_lens = sample_lens[mod_len:] # 将剩余句子的tokens数赋值给sample_lens;
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0 # 将剩余句子的最大tokens数赋值给sample_len;
        batch.append(idx)  # 若当前batch未满,则把当前句子示例的编号存到当前batch中,如[80,67,...];
        # 关于存在余数时,直接跳出_is_batch_full(),不将最后超出的那个句子添加到bacthes中,继续积累到batch进行下一次存储
    if len(batch) > 0:     # 处理最后一个batch,因其不能满足_is_batch_full,所以直接将其添加进batches中
        batches.append(batch)
    return batches  # 返回存储所有批次的列表,每个批次中有句子的编号
