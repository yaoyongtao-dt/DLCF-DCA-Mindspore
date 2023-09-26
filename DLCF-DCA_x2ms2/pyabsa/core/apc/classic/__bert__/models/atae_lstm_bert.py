# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.attention import NoQueryAttention
from ..layers.dynamic_rnn import DynamicLSTM
from ..layers.squeeze_embedding import SqueezeEmbedding
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class ATAE_LSTM_BERT(nn.Cell):
    inputs = ['text_indices', 'aspect_indices']

    def __init__(self, bert, opt):
        super(ATAE_LSTM_BERT, self).__init__()
        self.opt = opt
        self.embed = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim + opt.embed_dim, score_function='bi_linear')
        self.dense = x2ms_nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = x2ms_adapter.x2ms_sum(text_indices != 0, dim=-1)
        x_len_max = x2ms_adapter.x2ms_max(x_len)
        aspect_len = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1))

        x = self.embed(text_indices)['last_hidden_state']
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)['last_hidden_state']
        aspect_pool = x2ms_adapter.div(x2ms_adapter.x2ms_sum(aspect, dim=1), x2ms_adapter.tensor_api.unsqueeze(aspect_len, 1))
        aspect = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(aspect_pool, 1), -1, x_len_max, -1)
        x = x2ms_adapter.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = x2ms_adapter.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = x2ms_adapter.squeeze(x2ms_adapter.bmm(score, h), dim=1)

        out = self.dense(output)
        return out
