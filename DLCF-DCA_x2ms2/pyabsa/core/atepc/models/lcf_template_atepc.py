# -*- coding: utf-8 -*-
# file: lcf_template_atepc.py
# time: 2021/6/22
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import numpy as np
from transformers.models.bert.modeling_bert import BertForTokenClassification
import mindspore
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class LCF_TEMPLATE_ATEPC(BertForTokenClassification):

    def __init__(self, bert_base_model, opt):
        super(LCF_TEMPLATE_ATEPC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert4global = bert_base_model
        self.opt = opt
        self.bert4local = self.bert4global
        self.dropout = x2ms_nn.Dropout(self.opt.dropout)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(labels))
        for text_i in range(len(labels)):
            sep_index = x2ms_adapter.tensor_api.argmax(np, (labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return x2ms_adapter.to(x2ms_adapter.x2ms_tensor(labels), self.opt.device)

    def get_ids_for_local_context_extractor(self, text_indices):
        # convert BERT-SPC input to BERT-BASE format
        text_ids = x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(text_indices))
        for text_i in range(len(text_ids)):
            sep_index = x2ms_adapter.tensor_api.argmax(np, (text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return x2ms_adapter.to(x2ms_adapter.x2ms_tensor(text_ids), self.opt.device)

    def construct(self, input_ids_spc,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                polarity=None,
                valid_ids=None,
                attention_mask_label=None,
                lcf_cdm_vec=None,
                lcf_cdw_vec=None
                ):
        lcf_cdm_vec = x2ms_adapter.tensor_api.unsqueeze(lcf_cdm_vec, 2) if lcf_cdm_vec is not None else None
        lcf_cdw_vec = x2ms_adapter.tensor_api.unsqueeze(lcf_cdw_vec, 2) if lcf_cdw_vec is not None else None
        raise NotImplementedError('This is a template ATEPC model based on LCF, '
                                  'please implement your model use this template.')
