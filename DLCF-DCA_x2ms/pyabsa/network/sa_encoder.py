# -*- coding: utf-8 -*-
# file: sa_encoder.py
# time: 2021/6/6 0006
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import numpy as np
from ..core.apc.models.bert import AttentionLayer
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class Encoder(nn.Cell):
    def __init__(self, config, opt, layer_num=1):
        super(Encoder, self).__init__()
        self.opt = opt
        self.config = config
        self.encoder = x2ms_nn.ModuleList([SelfAttention(config, opt) for _ in range(layer_num)])
        self.tanh = x2ms_nn.Tanh()

    def construct(self, x):
        for i, enc in enumerate(self.encoder):
            x = x2ms_adapter.tensor_api.tanh(self, enc(x)[0])
        return x


class SelfAttention(nn.Cell):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = AttentionLayer(
            from_tensor_width=config.hidden_size,
            to_tensor_width=config.hidden_size,
            # from_seq_length=config.seq_length,  # 128
            # to_seq_length=config.seq_length,  # 128
            from_seq_length=128,
            to_seq_length=128,
            num_attention_heads=config.num_attention_heads,
            size_per_head=int(config.hidden_size / config.num_attention_heads),
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            has_attention_mask=True,
            do_return_2d_tensor=True)
        # compute_type=compute_type)

    def construct(self, inputs):
        zero_vec = np.zeros((x2ms_adapter.tensor_api.x2ms_size(inputs, 0), 1, 1, self.opt.max_seq_len))
        zero_tensor = x2ms_adapter.to(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_tensor(zero_vec)),
                                      inputs.device)
        SA_out = self.SA(inputs, zero_tensor)
        return SA_out
