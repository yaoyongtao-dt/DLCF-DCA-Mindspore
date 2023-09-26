# -*- coding: utf-8 -*-
# file: slide_lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class SLIDE_LCF_BERT(nn.Cell):
    inputs = ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec']

    def __init__(self, bert, opt):
        super(SLIDE_LCF_BERT, self).__init__()
        self.bert4global = bert
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)

        self.encoder = Encoder(bert.config, opt)
        self.encoder_left = Encoder(bert.config, opt)
        self.encoder_right = Encoder(bert.config, opt)

        self.post_linear = x2ms_nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_window_3h = x2ms_nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = x2ms_nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        self.post_encoder = Encoder(bert.config, opt)
        self.post_encoder_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_bert_indices = inputs[0]
        spc_mask_vec = inputs[1]
        lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[2], 2)
        left_lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[3], 2)
        right_lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[4], 2)

        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']
        masked_global_context_features = x2ms_adapter.mul(spc_mask_vec, global_context_features)

        # # --------------------------------------------------- #
        lcf_features = x2ms_adapter.mul(masked_global_context_features, lcf_matrix)
        lcf_features = self.encoder(lcf_features)
        # # --------------------------------------------------- #
        left_lcf_features = x2ms_adapter.mul(masked_global_context_features, left_lcf_matrix)
        left_lcf_features = self.encoder_left(left_lcf_features)
        # # --------------------------------------------------- #
        right_lcf_features = x2ms_adapter.mul(masked_global_context_features, right_lcf_matrix)
        right_lcf_features = self.encoder_right(right_lcf_features)
        # # --------------------------------------------------- #

        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            if self.opt.eta >= 0:
                cat_features = x2ms_adapter.cat(
                    (lcf_features, self.opt.eta * left_lcf_features, (1 - self.opt.eta) * right_lcf_features), -1)
            else:
                cat_features = x2ms_adapter.cat((lcf_features, left_lcf_features, right_lcf_features), -1)
            sent_out = self.linear_window_3h(cat_features)
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(x2ms_adapter.cat((lcf_features, left_lcf_features), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(x2ms_adapter.cat((lcf_features, right_lcf_features), -1))
        else:
            raise KeyError('Invalid parameter:', self.opt.window)

        sent_out = x2ms_adapter.cat((global_context_features, sent_out), -1)
        sent_out = self.post_linear(sent_out)
        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        dense_out = self.dense(sent_out)

        return dense_out
