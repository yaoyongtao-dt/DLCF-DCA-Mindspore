# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.attention import Attention
from ..layers.squeeze_embedding import SqueezeEmbedding
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class MemNet_BERT(nn.Cell):
    inputs = ['context_indices', 'aspect_indices']

    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = x2ms_adapter.tensor_api.numpy(memory_len)
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1 - float(idx + 1) / memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = x2ms_adapter.to(x2ms_adapter.x2ms_tensor(weight), self.opt.device)
        memory = x2ms_adapter.tensor_api.unsqueeze(weight, 2) * memory
        return memory

    def __init__(self, bert, opt):
        super(MemNet_BERT, self).__init__()
        self.opt = opt
        self.embed = bert
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = x2ms_nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = x2ms_adapter.x2ms_sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = x2ms_adapter.to(x2ms_adapter.x2ms_tensor(aspect_len, dtype=mindspore.float32), self.opt.device)

        memory = self.embed(text_raw_without_aspect_indices)['last_hidden_state']
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)['last_hidden_state']
        aspect = x2ms_adapter.x2ms_sum(aspect, dim=1)
        aspect = x2ms_adapter.div(aspect, x2ms_adapter.tensor_api.view(nonzeros_aspect, x2ms_adapter.tensor_api.x2ms_size(nonzeros_aspect, 0), 1))
        x = x2ms_adapter.tensor_api.unsqueeze(aspect, dim=1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x2ms_adapter.tensor_api.view(x, x2ms_adapter.tensor_api.x2ms_size(x, 0), -1)
        out = self.dense(x)
        return out
