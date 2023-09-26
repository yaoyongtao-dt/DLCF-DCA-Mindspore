# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.attention import Attention
from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class IAN_BERT(nn.Cell):
    inputs = ['text_indices', 'aspect_indices']

    def __init__(self, embedding_matrix, opt):
        super(IAN_BERT, self).__init__()
        self.opt = opt
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = x2ms_nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)['last_hidden_state']
        aspect = self.embed(aspect_indices)['last_hidden_state']
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = x2ms_adapter.to(x2ms_adapter.x2ms_tensor(aspect_len, dtype=mindspore.float32), self.opt.device)
        aspect_pool = x2ms_adapter.x2ms_sum(aspect, dim=1)
        aspect_pool = x2ms_adapter.div(aspect_pool, x2ms_adapter.tensor_api.view(aspect_len, x2ms_adapter.tensor_api.x2ms_size(aspect_len, 0), 1))

        text_raw_len = x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.clone(text_raw_len))
        context_pool = x2ms_adapter.x2ms_sum(context, dim=1)
        context_pool = x2ms_adapter.div(context_pool, x2ms_adapter.tensor_api.view(text_raw_len, x2ms_adapter.tensor_api.x2ms_size(text_raw_len, 0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = x2ms_adapter.tensor_api.squeeze(aspect_final, dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = x2ms_adapter.tensor_api.squeeze(context_final, dim=1)

        x = x2ms_adapter.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out
