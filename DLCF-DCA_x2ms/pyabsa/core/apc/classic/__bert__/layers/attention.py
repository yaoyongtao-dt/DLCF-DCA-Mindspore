# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class Attention(nn.Cell):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = x2ms_nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = x2ms_nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = x2ms_nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = x2ms_nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = mindspore.Parameter(x2ms_adapter.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = mindspore.Parameter(x2ms_adapter.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / x2ms_adapter.tensor_api.sqrt(math, self.hidden_dim)
        if self.weight is not None:
            x2ms_adapter.tensor_api.uniform_(self.weight.data, -stdv, stdv)

    def construct(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = x2ms_adapter.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = x2ms_adapter.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = x2ms_adapter.tensor_api.view(self.w_k(k), mb_size, k_len, self.n_head, self.hidden_dim)
        kx = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(kx, 2, 0, 1, 3)), -1, k_len, self.hidden_dim)
        qx = x2ms_adapter.tensor_api.view(self.w_q(q), mb_size, q_len, self.n_head, self.hidden_dim)
        qx = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(qx, 2, 0, 1, 3)), -1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = x2ms_adapter.tensor_api.permute(kx, 0, 2, 1)
            score = x2ms_adapter.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = x2ms_adapter.tensor_api.permute(kx, 0, 2, 1)
            qkt = x2ms_adapter.bmm(qx, kt)
            score = x2ms_adapter.div(qkt, x2ms_adapter.tensor_api.sqrt(math, self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = x2ms_adapter.tensor_api.expand(x2ms_adapter.unsqueeze(kx, dim=1), -1, q_len, -1, -1)
            qxx = x2ms_adapter.tensor_api.expand(x2ms_adapter.unsqueeze(qx, dim=2), -1, -1, k_len, -1)
            kq = x2ms_adapter.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = x2ms_adapter.nn_functional.tanh(x2ms_adapter.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = x2ms_adapter.matmul(qx, self.weight)
            kt = x2ms_adapter.tensor_api.permute(kx, 0, 2, 1)
            score = x2ms_adapter.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = x2ms_adapter.nn_functional.softmax(score, dim=-1)
        output = x2ms_adapter.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = x2ms_adapter.cat(x2ms_adapter.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1,
                 dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = mindspore.Parameter(x2ms_adapter.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / x2ms_adapter.tensor_api.sqrt(math, self.embed_dim)
        x2ms_adapter.tensor_api.uniform_(self.q.data, -stdv, stdv)

    def construct(self, k, **kwargs):
        mb_size = k.shape[0]
        q = x2ms_adapter.tensor_api.expand(self.q, mb_size, -1, -1)
        return x2ms_adapter.forward(super(NoQueryAttention, self), k, q)
