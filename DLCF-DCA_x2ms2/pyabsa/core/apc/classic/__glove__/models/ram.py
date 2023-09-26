# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class RAM(nn.Cell):
    inputs = ['text_indices', 'aspect_indices', 'left_indices']

    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = x2ms_adapter.tensor_api.numpy(memory_len)
        left_len = x2ms_adapter.tensor_api.numpy(left_len)
        aspect_len = x2ms_adapter.tensor_api.numpy(aspect_len)
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(1 - (idx - left_len[i] - aspect_len[i] + 1) / memory_len[i])
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.to(x2ms_adapter.x2ms_tensor(u, dtype=memory.dtype), self.opt.device), 2)
        weight = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.to(x2ms_adapter.x2ms_tensor(weight), self.opt.device), 2)
        v = memory * weight
        memory = x2ms_adapter.cat([v, u], dim=2)
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                           bidirectional=True)
        self.att_linear = x2ms_nn.Linear(opt.hidden_dim * 2 + 1 + opt.embed_dim * 2, 1)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2 + 1, opt.embed_dim)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs[0], inputs[1], inputs[2]
        left_len = x2ms_adapter.x2ms_sum(text_left_indices != 0, dim=-1)
        memory_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = x2ms_adapter.tensor_api.x2ms_float(aspect_len)

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        memory = x2ms_adapter.tensor_api.x2ms_float(memory)
        aspect = self.embed(aspect_indices)
        aspect = x2ms_adapter.x2ms_sum(aspect, dim=1)
        aspect = x2ms_adapter.div(aspect, x2ms_adapter.tensor_api.unsqueeze(nonzeros_aspect, -1))
        et = x2ms_adapter.to(x2ms_adapter.zeros_like(aspect), self.opt.device)

        batch_size = x2ms_adapter.tensor_api.x2ms_size(memory, 0)
        seq_len = x2ms_adapter.tensor_api.x2ms_size(memory, 1)
        for _ in range(self.opt.hops):
            g = self.att_linear(x2ms_adapter.cat([memory,
                                           x2ms_adapter.to(
                                               x2ms_adapter.zeros(batch_size, seq_len, self.opt.embed_dim), self.opt.device) + x2ms_adapter.tensor_api.unsqueeze(et, 1),
                                           x2ms_adapter.to(
                                               x2ms_adapter.zeros(batch_size, seq_len, self.opt.embed_dim), self.opt.device) + x2ms_adapter.tensor_api.unsqueeze(aspect, 1)],
                                          dim=-1))
            alpha = x2ms_adapter.nn_functional.softmax(g, dim=1)
            i = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.bmm(x2ms_adapter.tensor_api.transpose(alpha, 1, 2), memory), 1)
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out
