# -*- coding: utf-8 -*-
# @FileName: fast_lcf_bert_att.py
# @Time    : 2021/6/20 9:29
# @Author  : yangheng@m.scnu.edu.cn
# @github  : https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class FAST_LCF_BERT_ATT(nn.Cell):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec']

    def __init__(self, bert, opt):
        super(FAST_LCF_BERT_ATT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.bert_SA = Encoder(bert.config, opt)
        self.linear3 = x2ms_nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.bert_SA_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)
        print('{} is a test model!'.format(self.__class__.__name__))

    def construct(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs[0]
        else:
            text_bert_indices = inputs[1]
        text_local_indices = inputs[1]
        lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[2], 2)
        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']

        # LCF layer
        lcf_features = x2ms_adapter.mul(global_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        alpha_mat = x2ms_adapter.matmul(lcf_features, x2ms_adapter.tensor_api.transpose(global_context_features, 1, 2))
        alpha = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_sum(alpha_mat, 1, keepdim=True), dim=2)
        lcf_att_features = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(alpha, global_context_features), 1)  # batch_size x 2*hidden_dim
        global_features = self.bert_pooler(global_context_features)
        lcf_features = self.bert_pooler(lcf_features)
        out = self.linear3(x2ms_adapter.cat((global_features, lcf_att_features, lcf_features), dim=-1))

        dense_out = self.dense(out)
        return dense_out
