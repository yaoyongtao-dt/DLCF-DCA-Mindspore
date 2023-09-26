# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class SqueezeEmbedding(nn.Cell):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """

    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def construct(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.sort(-x_len)[1])
        x_unsort_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.sort(x_sort_idx)[1])
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = x2ms_nn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """unpack: out"""
        out = x2ms_nn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]  #
        """unsort"""
        out = out[x_unsort_idx]
        return out
