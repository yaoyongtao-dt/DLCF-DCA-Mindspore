# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/31 0031
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
import tqdm

from pyabsa.utils.pyabsa_utils import check_and_fix_labels
from .apc_utils import build_sentiment_window, build_spc_mask_vec, load_apc_datasets, prepare_input_for_apc, configure_spacy_model
from .apc_utils_for_dlcf_dca import prepare_input_for_dlcf_dca
import mindspore
import x2ms_adapter

# 这里是数据加载和处理
class ABSADataset:

    def __init__(self, fname, tokenizer, opt):
        configure_spacy_model(opt)
        ABSADataset.opt = opt

        lines = load_apc_datasets(fname)

        all_data = []
        # record polarities type to update polarities_dim
        label_set = set()

        for i in tqdm.tqdm(range(0, len(lines), 3), postfix='building word indices...'):
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity)
            x2ms_adapter.tensor_api.add(label_set, polarity)

            prepared_inputs = prepare_input_for_apc(opt, tokenizer, text_left, text_right, aspect)

            attention_mask = prepared_inputs['attention_mask']
            token_type_ids = prepared_inputs['token_type_ids']

            text_raw = prepared_inputs['text_raw']
            aspect = prepared_inputs['aspect']
            aspect_position = prepared_inputs['aspect_position']
            text_bert_indices = prepared_inputs['text_bert_indices']
            text_raw_bert_indices = prepared_inputs['text_raw_bert_indices']
            aspect_bert_indices = prepared_inputs['aspect_bert_indices']
            lcf_vec = prepared_inputs['lcf_cdm_vec'] if opt.lcf == 'cdm' else prepared_inputs['lcf_cdw_vec']
            if opt.model_name == 'dlcf_dca_bert':
                prepared_inputs = prepare_input_for_dlcf_dca(opt, tokenizer, text_left, text_right, aspect)
                dlcf_vec = prepared_inputs['dlcf_cdm_vec'] if opt.lcf == 'cdm' else prepared_inputs['dlcf_cdw_vec']
                depend_vec = prepared_inputs['depend_vec']
                depended_vec = prepared_inputs['depended_vec']
            data = {
                'ex_id': i // 3,

                'text_raw': text_raw,

                'aspect': aspect,

                'aspect_position': aspect_position,

                'lca_ids': lcf_vec,  # the lca indices are the same as the refactored CDM (lcf != CDW or Fusion) lcf vec

                'lcf_vec': lcf_vec if 'lcf_vec' in opt.model.inputs else 0,

                'dlcf_vec': dlcf_vec if 'dlcf_vec' in opt.model.inputs else 0,

                'spc_mask_vec': build_spc_mask_vec(opt, text_raw_bert_indices)
                if 'spc_mask_vec' in opt.model.inputs else 0,

                'text_bert_indices': text_bert_indices
                if 'text_bert_indices' in opt.model.inputs else 0,

                'aspect_bert_indices': aspect_bert_indices
                if 'aspect_bert_indices' in opt.model.inputs else 0,

                'text_raw_bert_indices': text_raw_bert_indices
                if 'text_raw_bert_indices' in opt.model.inputs else 0,

                'depend_vec': depend_vec if 'depend_vec' in opt.model.inputs else 0,

                'depended_vec': depended_vec if 'depended_vec' in opt.model.inputs else 0,

                'polarity': polarity,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids
            }

            x2ms_adapter.tensor_api.add(label_set, polarity)

            all_data.append(data)

        check_and_fix_labels(label_set, 'polarity', all_data, opt)
        opt.polarities_dim = len(label_set)

        if opt.model_name in ['slide_lcf_bert', 'slide_lcfs_bert', 'ssw_t', 'ssw_s']:
            all_data = build_sentiment_window(all_data, tokenizer, opt.similarity_threshold)
            for data in all_data:

                cluster_ids = []
                for pad_idx in range(opt.max_seq_len):
                    if pad_idx in data['cluster_ids']:
                        cluster_ids.append(data['polarity'])
                    else:
                        cluster_ids.append(-100)
                        # cluster_ids.append(3)

                data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
                data['side_ex_ids'] = np.array(0)
                data['aspect_position'] = np.array(0)

        else:
            for data in all_data:
                data['aspect_position'] = np.array(0)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
