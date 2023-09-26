# -*- coding: utf-8 -*-
# file: apc_utils.py
# time: 2021/5/23 0023
# author: xumayi <xumayi@m.scnu.edu.cn>
# github: https://github.com/XuMayi
# Copyright (C) 2021. All Rights Reserved.

# from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn

class BertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def weight_distrubute_local(bert_local_out, depend_weight, depended_weight, depend_vec, depended_vec, opt):
    bert_local_out2 = x2ms_adapter.zeros_like(bert_local_out)
    depend_vec2 = x2ms_adapter.mul(depend_vec, x2ms_adapter.tensor_api.unsqueeze(depend_weight, 2))
    depended_vec2 = x2ms_adapter.mul(depended_vec, x2ms_adapter.tensor_api.unsqueeze(depended_weight, 2))
    bert_local_out2 = bert_local_out2 + x2ms_adapter.mul(bert_local_out, depend_vec2) + x2ms_adapter.mul(bert_local_out, depended_vec2)
    for j in range(x2ms_adapter.tensor_api.x2ms_size(depend_weight)[0]):
         bert_local_out2[j][0] = bert_local_out[j][0]
    return bert_local_out2


class PointwiseFeedForward(nn.Cell):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = x2ms_nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = x2ms_nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = x2ms_nn.Dropout(dropout)
        self.relu = x2ms_nn.ReLU()

    def construct(self, x):
        output = self.relu(self.w_1(x2ms_adapter.tensor_api.transpose(x, 1, 2)))
        output = x2ms_adapter.tensor_api.transpose(self.w_2(output), 2, 1)
        output = self.dropout(output)
        return output


class DLCF_DCA_BERT(nn.Cell):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'dlcf_vec', 'depend_vec', 'depended_vec']

    def __init__(self, bert, opt):
        super(DLCF_DCA_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global

        self.hidden = opt.embed_dim
        self.opt = opt
        self.opt.bert_dim = opt.embed_dim
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.bert_SA_ = Encoder(bert.config, opt)

        self.mean_pooling_double = PointwiseFeedForward(self.hidden * 2, self.hidden, self.hidden)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(self.hidden, opt.polarities_dim)

        self.dca_sa = x2ms_nn.ModuleList()
        self.dca_pool = x2ms_nn.ModuleList()
        self.dca_lin = x2ms_nn.ModuleList()

        for i in range(opt.dca_layer):
            self.dca_sa.append(Encoder(bert.config, opt))
            self.dca_pool.append(BertPooler(bert.config))
            self.dca_lin.append(x2ms_nn.Sequential(
                x2ms_nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                x2ms_nn.GELU(),
                x2ms_nn.Linear(opt.bert_dim * 2, 1),
                x2ms_nn.Sigmoid())
            )

    def weight_calculate(self, sa, pool, lin, d_w, ded_w, depend_out, depended_out):
        depend_sa_out = sa(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out = sa(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = pool(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = pool(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = lin(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = lin(depended_pool_out)
        depended_weight = self.dropout(depended_weight)

        for i in range(x2ms_adapter.tensor_api.x2ms_size(depend_weight)[0]):
            depend_weight[i] = x2ms_adapter.tensor_api.item(depend_weight[i]) * x2ms_adapter.tensor_api.item(d_w[i])
            depended_weight[i] = x2ms_adapter.tensor_api.item(depended_weight[i]) * x2ms_adapter.tensor_api.item(ded_w[i])
            weight_sum = x2ms_adapter.tensor_api.item(depend_weight[i]) + x2ms_adapter.tensor_api.item(depended_weight[i])
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.dca_p
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.dca_p
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight

    def construct(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs[0]
        else:
            text_bert_indices = inputs[1]
        text_local_indices = inputs[1]
        lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(inputs[2], 2)
        depend_vec = x2ms_adapter.tensor_api.unsqueeze(inputs[3], 2)
        depended_vec = x2ms_adapter.tensor_api.unsqueeze(inputs[4], 2)

        global_context_features,_ = self.bert4global(text_bert_indices)
        local_context_features,_ = self.bert4local(text_local_indices)

        bert_local_out = x2ms_adapter.mul(local_context_features, lcf_matrix)

        depend_weight = x2ms_adapter.ones(x2ms_adapter.tensor_api.x2ms_size(bert_local_out)[0])
        depended_weight = x2ms_adapter.ones(x2ms_adapter.tensor_api.x2ms_size(bert_local_out)[0])

        for i in range(self.opt.dca_layer):
            depend_out = x2ms_adapter.mul(bert_local_out,depend_vec)
            depended_out = x2ms_adapter.mul(bert_local_out,depended_vec)
            depend_weight, depended_weight = self.weight_calculate(self.dca_sa[i], self.dca_pool[i], self.dca_lin[i],
                                                                   depend_weight, depended_weight, depend_out,
                                                                   depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out, depend_weight, depended_weight, depend_vec, depended_vec,
                                                      self.opt)

        out_cat = x2ms_adapter.cat((bert_local_out, global_context_features), dim=-1)
        out_cat = self.mean_pooling_double(out_cat)
        out_cat = self.bert_SA_(out_cat)
        out_cat = self.bert_pooler(out_cat)
        dense_out = self.dense(out_cat)
        return dense_out