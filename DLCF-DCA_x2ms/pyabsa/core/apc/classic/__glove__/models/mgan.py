# -*- coding: utf-8 -*-
# file: mgan.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class LocationEncoding(nn.Cell):
    def __init__(self, opt):
        super(LocationEncoding, self).__init__()
        self.opt = opt

    def construct(self, x, pos_inx):
        batch_size, seq_len = x2ms_adapter.tensor_api.x2ms_size(x)[0], x2ms_adapter.tensor_api.x2ms_size(x)[1]
        weight = x2ms_adapter.to(self.weight_matrix(pos_inx, batch_size, seq_len), self.opt.device)
        x = x2ms_adapter.tensor_api.unsqueeze(weight, 2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = x2ms_adapter.tensor_api.numpy(pos_inx)
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][0]):
                relative_pos = pos_inx[i][0] - j
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            for j in range(pos_inx[i][0], pos_inx[i][1] + 1):
                weight[i].append(0)
            for j in range(pos_inx[i][1] + 1, seq_len):
                relative_pos = j - pos_inx[i][1]
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
        weight = x2ms_adapter.x2ms_tensor(weight)
        return weight


class AlignmentMatrix(nn.Cell):
    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = mindspore.Parameter(x2ms_adapter.Tensor(6 * opt.hidden_dim, 1))

    def construct(self, batch_size, ctx, asp):
        ctx_len = x2ms_adapter.tensor_api.x2ms_size(ctx, 1)
        asp_len = x2ms_adapter.tensor_api.x2ms_size(asp, 1)
        alignment_mat = x2ms_adapter.to(x2ms_adapter.zeros(batch_size, ctx_len, asp_len), self.opt.device)
        ctx_chunks = x2ms_adapter.tensor_api.chunk(ctx, ctx_len, dim=1)
        asp_chunks = x2ms_adapter.tensor_api.chunk(asp, asp_len, dim=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = x2ms_adapter.cat([ctx_chunk, asp_chunk, ctx_chunk * asp_chunk], dim=2)  # batch_size x 1 x 6*hidden_dim
                alignment_mat[:, i, j] = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.matmul(feat, x2ms_adapter.tensor_api.expand(self.w_u, batch_size, -1, -1)), -1), -1)
        return alignment_mat


class MGAN(nn.Cell):
    inputs = ['text_indices', 'aspect_indices', 'left_indices']

    def __init__(self, embedding_matrix, opt):
        super(MGAN, self).__init__()
        self.opt = opt
        self.embed = x2ms_nn.Embedding.from_pretrained(x2ms_adapter.x2ms_tensor(embedding_matrix, dtype=mindspore.float32))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.location = LocationEncoding(opt)
        self.w_a2c = mindspore.Parameter(x2ms_adapter.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.w_c2a = mindspore.Parameter(x2ms_adapter.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.alignment = AlignmentMatrix(opt)
        self.dense = x2ms_nn.Linear(8 * opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices = inputs[0]  # batch_size x seq_len
        aspect_indices = inputs[1]
        text_left_indices = inputs[2]
        batch_size = x2ms_adapter.tensor_api.x2ms_size(text_raw_indices, 0)
        ctx_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=1)
        asp_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=1)
        left_len = x2ms_adapter.x2ms_sum(text_left_indices != 0, dim=-1)
        aspect_in_text = x2ms_adapter.cat([x2ms_adapter.tensor_api.unsqueeze(left_len, -1), x2ms_adapter.tensor_api.unsqueeze((left_len + asp_len - 1), -1)], dim=-1)

        ctx = self.embed(text_raw_indices)  # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices)  # batch_size x seq_len x embed_dim

        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len)
        ctx_out = x2ms_adapter.tensor_api.x2ms_float(self.location(ctx_out, aspect_in_text))  # batch_size x (ctx)seq_len x 2*hidden_dim
        ctx_pool = x2ms_adapter.x2ms_sum(ctx_out, dim=1)
        ctx_pool = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.div(ctx_pool, x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.x2ms_float(ctx_len), -1)), -1)  # batch_size x 2*hidden_dim x 1

        asp_out, (_, _) = self.asp_lstm(asp, asp_len)  # batch_size x (asp)seq_len x 2*hidden_dim
        asp_pool = x2ms_adapter.x2ms_sum(asp_out, dim=1)
        asp_pool = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.div(asp_pool, x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.x2ms_float(asp_len), -1)), -1)  # batch_size x 2*hidden_dim x 1

        alignment_mat = self.alignment(batch_size, ctx_out, x2ms_adapter.tensor_api.x2ms_float(asp_out))  # batch_size x (ctx)seq_len x (asp)seq_len
        # batch_size x 2*hidden_dim
        f_asp2ctx = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(x2ms_adapter.tensor_api.transpose(ctx_out, 1, 2),
                                 x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_max(alignment_mat, 2, keepdim=True)[0], dim=1)), -1)
        f_ctx2asp = x2ms_adapter.tensor_api.squeeze(
            x2ms_adapter.tensor_api.transpose(x2ms_adapter.matmul(x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_max(alignment_mat, 1, keepdim=True)[0], dim=2), asp_out), 1,
                                                                                                             2), -1)

        c_asp2ctx_alpha = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.matmul(x2ms_adapter.tensor_api.matmul(ctx_out, x2ms_adapter.tensor_api.expand(self.w_a2c, batch_size, -1, -1)), asp_pool), dim=1)
        c_asp2ctx = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(x2ms_adapter.tensor_api.transpose(ctx_out, 1, 2), c_asp2ctx_alpha), -1)
        c_ctx2asp_alpha = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.matmul(x2ms_adapter.tensor_api.matmul(asp_out, x2ms_adapter.tensor_api.expand(self.w_c2a, batch_size, -1, -1)), ctx_pool), dim=1)
        c_ctx2asp = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.matmul(x2ms_adapter.tensor_api.transpose(asp_out, 1, 2), c_ctx2asp_alpha), -1)

        feat = x2ms_adapter.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], dim=1)
        out = self.dense(feat)  # batch_size x polarity_dim

        return out
