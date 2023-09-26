# -*- coding: utf-8 -*-
# file: tc_lstm.py
# author: songyouwei <youwei0314@gmail.com>, ZhangYikai <zykhelloha@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class TC_LSTM_BERT(nn.Cell):
    inputs = ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices']

    def __init__(self, bert, opt):
        super(TC_LSTM_BERT, self).__init__()
        self.embed = bert
        self.lstm_l = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = x2ms_nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def construct(self, inputs):
        # Get the target and its length(target_len)
        x_l, x_r, target = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = x2ms_adapter.x2ms_sum(x_l != 0, dim=-1), x2ms_adapter.x2ms_sum(x_r != 0, dim=-1)
        target_len = x2ms_adapter.x2ms_sum(target != 0, dim=-1, dtype=mindspore.float32)[:, None, None]
        x_l, x_r, target = self.embed(x_l)['last_hidden_state'], self.embed(x_r)['last_hidden_state'], self.embed(target)['last_hidden_state']
        v_target = x2ms_adapter.div(x2ms_adapter.tensor_api.x2ms_sum(target, dim=1, keepdim=True),
                             target_len)  # v_{target} in paper: average the target words

        # the concatenation of word embedding and target vector v_{target}:
        x_l = x2ms_adapter.cat(
            (x_l, x2ms_adapter.cat(([v_target] * x_l.shape[1]), 1)),
            2
        )
        x_r = x2ms_adapter.cat(
            (x_r, x2ms_adapter.cat(([v_target] * x_r.shape[1]), 1)),
            2
        )

        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = x2ms_adapter.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
