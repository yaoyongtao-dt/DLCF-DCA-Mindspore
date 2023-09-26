# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class BERT_SPC(nn.Cell):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_bert_indices = inputs[0]
        _,pooled_output = self.bert(text_bert_indices)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
