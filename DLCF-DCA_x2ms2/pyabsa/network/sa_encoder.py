# -*- coding: utf-8 -*-
# file: sa_encoder.py
# time: 2021/6/6 0006
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import numpy as np
# from transformers.models.bert.modeling_bert import BertSelfAttention
# 这里改过
from mindnlp.models.bert.bert import BertSelfAttention
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
        self.SA = BertSelfAttention(config)

    def construct(self, inputs):
        zero_vec = np.zeros((x2ms_adapter.tensor_api.x2ms_size(inputs, 0), 1, 1, self.opt.max_seq_len))
        zero_tensor = x2ms_adapter.to(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_tensor(zero_vec)), inputs.device)
        SA_out = self.SA(inputs, zero_tensor)
        return SA_out
