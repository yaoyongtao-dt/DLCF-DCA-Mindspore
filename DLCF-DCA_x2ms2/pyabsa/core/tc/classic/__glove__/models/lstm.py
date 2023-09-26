# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class LSTM(nn.Cell):
    inputs = ['text_indices']

    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = x2ms_nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out
