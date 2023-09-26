# -*- coding: utf-8 -*-
# file: bert_base.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

from transformers.models.bert.modeling_bert import BertPooler
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


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
