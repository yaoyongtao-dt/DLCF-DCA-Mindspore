# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import numpy as np
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
from mindspore import Tensor
import mindspore.common.dtype as mstype

class BERT_SPC(nn.Cell):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(p=opt.dropout)
        self.dense = nn.Dense(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        # if isinstance(inputs, list):
        #     text_bert_indices = inputs[0]
        # else:
        #     text_bert_indices = inputs
        # target_length = 128
        # padding_length = target_length - text_bert_indices.shape[1]
        # pad_op = mindspore.ops.Pad(paddings=((0, 0), (0, padding_length)))
        # text_bert_indices = pad_op(text_bert_indices)
        #
        # # text_bert_indices = text_bert_indices[0:1, :]
        #
        # # 假设你的输入文本序列的长度为 seq_length
        # seq_length = text_bert_indices.shape[1]
        #
        split_tensors = np.split(inputs, 3, axis=1)
        # # 创建一个全1的输入掩码（input_mask），表示所有位置都是有效的标记
        input_mask = np.ones((split_tensors[0].shape[0], self.opt.max_seq_len), dtype=np.int32)
        input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int32)
        #
        # # 创建一个全0的标记类型标识符（token_type_ids），因为没有涉及多句子任务
        token_type_ids = np.zeros((split_tensors[0].shape[0], self.opt.max_seq_len), dtype=np.int32)
        token_type_ids = mindspore.Tensor(token_type_ids, dtype=mindspore.int32)
        # masked_lm_positions = np.zeros((text_bert_indices.shape[0], 1), dtype=np.int32)
        # masked_lm_positions = mindspore.Tensor(masked_lm_positions, dtype=mindspore.int32)
        # self.bert.set_train(True)
        self.bert.is_training = True
        out = self.bert.bert_forward(split_tensors[0], input_mask, token_type_ids)
        pooled_output = out[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)



        # print(logits)
        # # 伪向量测试
        # values = [0.1, 0.2, 0.3]
        # print(self.bert.is_training)
        # 创建一个空列表来存储 n 个[1, 2, 3]
        # tensor_list = []
        # for _ in range(text_bert_indices.shape[0]):
        #     tensor_list.append(values)
        #
        # # 使用 np.array 将列表转换为 NumPy 数组
        # tensor_array = np.array(tensor_list)
        #
        # # 将 NumPy 数组转换为 MindSpore 张量
        # logits = mindspore.Tensor(tensor_array, dtype=mindspore.float32)

        # 输出张量的形状
        # print(logits.shape)

        # 输出结果
        # print(logits)
        # 测试
        # logits = x2ms_adapter.softmax(logits,-1)

        # print(logits)
        logits.set_dtype(mindspore.float32)
        # print(logits)
        return logits
