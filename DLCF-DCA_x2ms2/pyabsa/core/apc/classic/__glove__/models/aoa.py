# -*- coding: utf-8 -*-
# file: aoa.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class AOA(nn.Cell):
    inputs = ['text_indices', 'aspect_indices']

    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = x2ms_nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_indices = inputs[0]  # batch_size x seq_len
        aspect_indices = inputs[1]  # batch_size x seq_len
        ctx_len = x2ms_adapter.x2ms_sum(text_indices != 0, dim=1)
        asp_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_indices)  # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices)  # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len)  # batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)  # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = x2ms_adapter.matmul(ctx_out,
                                       x2ms_adapter.transpose(asp_out, 1, 2))  # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = x2ms_adapter.nn_functional.softmax(interaction_mat, dim=1)  # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = x2ms_adapter.nn_functional.softmax(interaction_mat, dim=2)  # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = x2ms_adapter.tensor_api.mean(beta, dim=1, keepdim=True)  # batch_size x 1 x (asp) seq_len
        gamma = x2ms_adapter.matmul(alpha, x2ms_adapter.tensor_api.transpose(beta_avg, 1, 2))  # batch_size x (ctx) seq_len x 1
        weighted_sum = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(x2ms_adapter.transpose(ctx_out, 1, 2), gamma), -1)  # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum)  # batch_size x polarity_dim

        return out
