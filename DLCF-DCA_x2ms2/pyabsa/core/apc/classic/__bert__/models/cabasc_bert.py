# -*- coding: utf-8 -*-
# file: cabasc.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from ..layers.dynamic_rnn import DynamicLSTM
from ..layers.squeeze_embedding import SqueezeEmbedding
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class Cabasc_BERT(nn.Cell):
    inputs = ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices']

    def __init__(self, bert, opt, _type='c'):
        super(Cabasc_BERT, self).__init__()
        self.opt = opt
        self.type = _type
        self.embed = bert
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.linear1 = x2ms_nn.Linear(3 * opt.embed_dim, opt.embed_dim)
        self.linear2 = x2ms_nn.Linear(opt.embed_dim, 1, bias=False)
        self.mlp = x2ms_nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)
        # context attention layer
        self.rnn_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type='GRU')
        self.rnn_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type='GRU')
        self.mlp_l = x2ms_nn.Linear(opt.hidden_dim, 1)
        self.mlp_r = x2ms_nn.Linear(opt.hidden_dim, 1)

    def context_attention(self, x_l, x_r, memory, memory_len, aspect_len):

        # Context representation
        left_len, right_len = x2ms_adapter.x2ms_sum(x_l != 0, dim=-1), x2ms_adapter.x2ms_sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l)['last_hidden_state'], self.embed(x_r)['last_hidden_state']

        context_l, (_, _) = self.rnn_l(x_l, left_len)  # left, right context : (batch size, max_len, embedds)
        context_r, (_, _) = self.rnn_r(x_r, right_len)

        # Attention weights : (batch_size, max_batch_len, 1) 
        # 0.5 should be a variable according to the paper
        attn_l = x2ms_adapter.sigmoid(self.mlp_l(context_l)) + 0.5
        attn_r = x2ms_adapter.sigmoid(self.mlp_r(context_r)) + 0.5

        # apply weights one sample at a time
        for i in range(x2ms_adapter.tensor_api.x2ms_size(memory, 0)):
            aspect_start = x2ms_adapter.tensor_api.item((left_len[i] - aspect_len[i]))
            aspect_end = left_len[i]
            # attention weights for each element in the sentence
            for idx in range(memory_len[i]):
                if idx < aspect_start:
                    memory[i][idx] *= attn_l[i][idx]
                elif idx < aspect_end:
                    memory[i][idx] *= (attn_l[i][idx] + attn_r[i][idx - aspect_start]) / 2
                else:
                    memory[i][idx] *= attn_r[i][idx - aspect_start]

        return memory

    def locationed_memory(self, memory, memory_len):
        # based on the absolute distance to the first border word of the aspect
        '''
        # differ from description in paper here, but may be better
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                aspect_end = left_len[i] 
                if idx < aspect_start: l = aspect_start.item() - idx                   
                elif idx <= aspect_end: l = 0 
                else: l = idx - aspect_end.item()
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
        '''
        for i in range(x2ms_adapter.tensor_api.x2ms_size(memory, 0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1 - float(idx) / int(memory_len[i]))

        return memory

    def construct(self, inputs):
        # inputs
        text_raw_indices, aspect_indices, x_l, x_r = inputs[0], inputs[1], inputs[2], inputs[3]
        memory_len = x2ms_adapter.x2ms_sum(text_raw_indices != 0, dim=-1)
        aspect_len = x2ms_adapter.x2ms_sum(aspect_indices != 0, dim=-1)

        # aspect representation
        nonzeros_aspect = x2ms_adapter.tensor_api.x2ms_float(aspect_len)
        aspect = self.embed(aspect_indices)['last_hidden_state']
        aspect = x2ms_adapter.x2ms_sum(aspect, dim=1)
        v_a = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.div(aspect, x2ms_adapter.tensor_api.unsqueeze(nonzeros_aspect, 1)), 1)  # batch_size x 1 x embed_dim

        # memory module
        memory = self.embed(text_raw_indices)['last_hidden_state']
        memory = self.squeeze_embedding(memory, memory_len)

        # sentence representation 
        nonzeros_memory = x2ms_adapter.tensor_api.x2ms_float(memory_len)
        v_s = x2ms_adapter.x2ms_sum(memory, dim=1)
        v_s = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.div(v_s, x2ms_adapter.tensor_api.unsqueeze(nonzeros_memory, 1)), 1)  # batch_size x 1 x embed_dim

        # position attention module
        if self.type == 'c':
            memory = self.locationed_memory(memory, memory_len)  # batch_size x seq_len x embed_dim
        elif self.type == 'cabasc':
            # context attention
            memory = self.context_attention(x_l, x_r, memory, memory_len, aspect_len)
            # recalculate sentence rep with new memory
            v_s = x2ms_adapter.x2ms_sum(memory, dim=1)
            v_s = x2ms_adapter.div(v_s, x2ms_adapter.tensor_api.unsqueeze(nonzeros_memory, 1))
            v_s = x2ms_adapter.tensor_api.unsqueeze(v_s, dim=1)

        '''
        # no multi-hop, but may be better. 
        # however, here is something totally different from what paper depits
        for _ in range(self.opt.hops):  
            #x = self.x_linear(x)
            v_ts, _ = self.attention(memory, v_a)
        '''
        memory_chunks = x2ms_adapter.tensor_api.chunk(memory, x2ms_adapter.tensor_api.x2ms_size(memory, 1), dim=1)
        c = []
        for memory_chunk in memory_chunks:  # batch_size x 1 x embed_dim
            c_i = self.linear1(x2ms_adapter.tensor_api.view(x2ms_adapter.cat([memory_chunk, v_a, v_s], dim=1), x2ms_adapter.tensor_api.x2ms_size(memory_chunk, 0), -1))
            c_i = self.linear2(x2ms_adapter.tanh(c_i))  # batch_size x 1
            c.append(c_i)
        alpha = x2ms_adapter.nn_functional.softmax(x2ms_adapter.cat(c, dim=1), dim=1)  # batch_size x seq_len
        v_ts = x2ms_adapter.tensor_api.transpose(x2ms_adapter.matmul(x2ms_adapter.tensor_api.transpose(memory, 1, 2), x2ms_adapter.tensor_api.unsqueeze(alpha, -1)), 1, 2)

        # classifier
        v_ns = v_ts + v_s  # embedd the sentence
        v_ns = x2ms_adapter.tensor_api.view(v_ns, x2ms_adapter.tensor_api.x2ms_size(v_ns, 0), -1)
        v_ms = x2ms_adapter.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)

        return out
