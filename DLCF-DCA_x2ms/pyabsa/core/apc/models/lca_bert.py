# -*- coding: utf-8 -*-
# file: lca_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import copy
# from mindformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.loss as loss_wrapper
import x2ms_adapter.nn as x2ms_nn

class BertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LCA_BERT(nn.Cell):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec', 'polarity']

    def __init__(self, bert, opt):
        super(LCA_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert)
        self.lc_embed = x2ms_nn.Embedding(2, opt.embed_dim)
        self.lc_linear = x2ms_nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.opt = opt
        self.dropout = x2ms_nn.Dropout(opt.dropout)
        self.bert_SA_L = Encoder(bert.config, opt)
        self.bert_SA_G = Encoder(bert.config, opt)
        self.cat_linear = x2ms_nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.pool = BertPooler(bert.config)
        self.dense = x2ms_nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.classifier = x2ms_nn.Linear(opt.embed_dim, 2)
        self.lca_criterion = loss_wrapper.CrossEntropyLoss()
        self.classification_criterion = loss_wrapper.CrossEntropyLoss()

    def construct(self, inputs):
        if self.opt.use_bert_spc:
            text_global_indices = inputs[0]
        else:
            text_global_indices = inputs[1]
        text_local_indices = inputs[1]
        lca_ids = inputs[2]
        lcf_matrix = x2ms_adapter.tensor_api.unsqueeze(lca_ids, 2)  # lca_ids is the same as lcf_matrix
        polarity = inputs[3]

        bert_global_out = self.bert4global(text_global_indices)['last_hidden_state']
        bert_local_out = self.bert4local(text_local_indices)['last_hidden_state']

        lc_embedding = self.lc_embed(lca_ids)
        bert_global_out = self.lc_linear(x2ms_adapter.cat((bert_global_out, lc_embedding), -1))

        # # LCF-layer
        bert_local_out = x2ms_adapter.mul(bert_local_out, lcf_matrix)
        bert_local_out = self.bert_SA_L(bert_local_out)

        cat_features = x2ms_adapter.cat((bert_local_out, bert_global_out), dim=-1)
        cat_features = self.cat_linear(cat_features)

        lca_logits = self.classifier(cat_features)
        lca_logits = x2ms_adapter.tensor_api.view(lca_logits, -1, 2)
        lca_ids = x2ms_adapter.tensor_api.view(lca_ids, -1)

        cat_features = self.dropout(cat_features)

        pooled_out = self.pool(cat_features)
        sent_logits = self.dense(pooled_out)

        lcp_loss = self.lca_criterion(lca_logits, lca_ids)
        sent_loss = self.classification_criterion(sent_logits, polarity)

        return {'logits': sent_logits, 'loss': (1 - self.opt.sigma) * sent_loss + self.opt.sigma * lcp_loss}
