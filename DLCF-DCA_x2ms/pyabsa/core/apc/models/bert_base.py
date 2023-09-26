# -*- coding: utf-8 -*-
# file: bert_base.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

# from mindformers.models.bert import BertPooler
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn

class BertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERT_BASE(nn.Cell):
    inputs = ['text_raw_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_BASE, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_bert_indices = inputs[0]
        text_features = self.bert(text_bert_indices)['last_hidden_state']
        pooled_output = self.pooler(text_features)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
