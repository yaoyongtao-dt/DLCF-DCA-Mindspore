# -*- coding: utf-8 -*-
# file: lcf_pooler.py
# time: 2021/6/29
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class LCF_Pooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = x2ms_nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = x2ms_nn.Tanh()

    def construct(self, hidden_states, lcf_vec):
        device = hidden_states.device
        lcf_vec = x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(lcf_vec))

        pooled_output = numpy.zeros((hidden_states.shape[0], hidden_states.shape[2]), dtype=numpy.float32)
        hidden_states = x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(hidden_states))
        for i, vec in enumerate(lcf_vec):
            lcf_ids = [j for j in range(len(vec)) if sum(vec[j] - 1.) == 0]
            pooled_output[i] = hidden_states[i][lcf_ids[len(lcf_ids) // 2]]

        pooled_output = x2ms_adapter.to(x2ms_adapter.Tensor(pooled_output), device)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
