# -*- coding: utf-8 -*-
# file: asgcn.py
# author:  <gene_zhangchen@163.com>
# Copyright (C) 2020. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class GraphConvolution(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mindspore.Parameter(x2ms_adapter.FloatTensor(in_features, out_features))
        if bias:
            self.bias = mindspore.Parameter(x2ms_adapter.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def construct(self, text, adj):
        hidden = x2ms_adapter.matmul(text, self.weight)
        denom = x2ms_adapter.x2ms_sum(adj, dim=2, keepdim=True) + 1
        output = x2ms_adapter.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASGCN(nn.Cell):
    inputs = ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph']

    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = x2ms_nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = x2ms_nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = x2ms_adapter.tensor_api.numpy(aspect_double_idx)
        text_len = x2ms_adapter.tensor_api.numpy(text_len)
        aspect_len = x2ms_adapter.tensor_api.numpy(aspect_len)
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(max(0, 1 - (aspect_double_idx[i, 0] - j) / context_len))
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(max(0, 1 - (j - aspect_double_idx[i, 1]) / context_len))
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = x2ms_adapter.to(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.x2ms_tensor(weight, dtype=mindspore.float32), 2), self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = x2ms_adapter.tensor_api.numpy(aspect_double_idx)
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = x2ms_adapter.to(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.x2ms_tensor(mask, dtype=mindspore.float32), 2), self.opt.device)
        return mask * x

    def construct(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = x2ms_adapter.x2ms_sum(text_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)
        left_len = x2ms_adapter.x2ms_sum(left_indices != 0, dim=-1)
        aspect_double_idx = x2ms_adapter.cat([x2ms_adapter.tensor_api.unsqueeze(left_len, 1), x2ms_adapter.tensor_api.unsqueeze((left_len + aspect_len - 1), 1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = x2ms_adapter.nn_functional.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = x2ms_adapter.nn_functional.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = x2ms_adapter.matmul(x, x2ms_adapter.tensor_api.transpose(text_out, 1, 2))
        alpha = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_sum(alpha_mat, 1, keepdim=True), dim=2)
        x = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(alpha, text_out), 1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
