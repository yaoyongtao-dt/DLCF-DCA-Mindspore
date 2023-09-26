# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class LCF_BERT(nn.Cell):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec']

    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.bert_SA = Encoder(bert.config, opt)
        self.linear2 = x2ms_nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.bert_SA_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs[0]
        else:
            text_bert_indices = inputs[1]
        text_local_indices = inputs[1]
        lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[2], 2)
        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']
        local_context_features = self.bert4local(text_local_indices)['last_hidden_state']

        # LCF layer
        lcf_features = x2ms_adapter.mul(local_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        cat_features = x2ms_adapter.cat((lcf_features, global_context_features), dim=-1)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)
        cat_features = self.bert_SA_(cat_features)
        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)
        return dense_out
