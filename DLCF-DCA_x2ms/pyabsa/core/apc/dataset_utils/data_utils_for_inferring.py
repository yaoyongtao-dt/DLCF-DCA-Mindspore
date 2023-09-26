# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
from tqdm import tqdm

from .apc_utils import (build_sentiment_window,
                        build_spc_mask_vec,
                        load_apc_datasets,
                        prepare_input_for_apc,
                        LABEL_PADDING, configure_spacy_model)
from .apc_utils_for_dlcf_dca import prepare_input_for_dlcf_dca
import mindspore
import x2ms_adapter


class ABSADataset:

    def __init__(self, tokenizer, opt):
        configure_spacy_model(opt)
        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        _text = text
        samples = []
        try:
            if '!sent!' not in text:
                splits = x2ms_adapter.tensor_api.split(text, '[ASP]')
                for i in range(0, len(splits) - 1, 2):
                    sample = text.replace('[ASP]', '').replace(splits[i + 1], '[ASP]' + splits[i + 1] + '[ASP]')
                    samples.append(sample)
            else:
                text, ref_sent = x2ms_adapter.tensor_api.split(text, '!sent!')
                ref_sent = x2ms_adapter.tensor_api.split(ref_sent)
                text = '[PADDING] ' + text + ' [PADDING]'
                splits = x2ms_adapter.tensor_api.split(text, '[ASP]')

                if int((len(splits) - 1) / 2) == len(ref_sent):
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        sample += ' !sent! ' + str(ref_sent[int(i / 2)])
                        samples.append(sample.replace('[TEMP]', '[ASP]'))
                else:
                    print(_text,
                          ' -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.')
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        samples.append(sample.replace('[TEMP]', '[ASP]'))

        except:
            print('Invalid Input:', _text)
        return samples

    def prepare_infer_sample(self, text: str):
        self.process_data(self.parse_sample(text))

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_apc_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        label_set = set()
        for ex_id, text in enumerate(tqdm(samples, postfix='building word indices...')):
            try:
                # handle for empty lines in inferring_tutorials dataset_utils
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '!sent!' in text:
                    text, polarity = x2ms_adapter.tensor_api.split(text, '!sent!')[0].strip(), x2ms_adapter.tensor_api.split(text, '!sent!')[1].strip()
                    polarity = int(polarity) if polarity else LABEL_PADDING
                    text = text.replace('[PADDING]', '')

                    if polarity < 0:
                        raise RuntimeError(
                            'Invalid sentiment label detected, only please label the sentiment between {0, N-1} '
                            '(assume there are N types of sentiment polarities.)')
                else:
                    polarity = LABEL_PADDING

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = x2ms_adapter.tensor_api.split(text, '[ASP]')
                text_left = text_left.replace('[PADDING] ', '')
                text_right = text_right.replace(' [PADDING]', '')

                prepared_inputs = prepare_input_for_apc(self.opt, self.tokenizer, text_left, text_right, aspect)

                text_raw = prepared_inputs['text_raw']
                aspect = prepared_inputs['aspect']
                aspect_position = prepared_inputs['aspect_position']
                text_bert_indices = prepared_inputs['text_bert_indices']
                text_raw_bert_indices = prepared_inputs['text_raw_bert_indices']
                aspect_bert_indices = prepared_inputs['aspect_bert_indices']
                lcf_vec = prepared_inputs['lcf_cdm_vec'] if self.opt.lcf == 'cdm' else prepared_inputs['lcf_cdw_vec']

                if self.opt.model_name == 'dlcf_dca_bert':
                    prepared_inputs = prepare_input_for_dlcf_dca(self.opt, self.opt.tokenizer, text_left, text_right, aspect)
                    dlcf_vec = prepared_inputs['dlcf_cdm_vec'] if self.opt.lcf == 'cdm' else prepared_inputs['dlcf_cdw_vec']
                    depend_vec = prepared_inputs['depend_vec']
                    depended_vec = prepared_inputs['depended_vec']
                data = {
                    'ex_id': ex_id,

                    'depend_vec': depend_vec if 'depend_vec' in self.opt.model.inputs else 0,

                    'depended_vec': depended_vec if 'depended_vec' in self.opt.model.inputs else 0,

                    'text_raw': text_raw,

                    'aspect': aspect,

                    'aspect_position': aspect_position,

                    'lca_ids': lcf_vec,  # the lca indices are the same as the refactored CDM (lcf != CDW or Fusion) lcf vec

                    'lcf_vec': lcf_vec if 'lcf_vec' in self.opt.model.inputs else 0,

                    'dlcf_vec': dlcf_vec if 'dlcf_vec' in self.opt.model.inputs else 0,

                    'spc_mask_vec': build_spc_mask_vec(self.opt, text_raw_bert_indices)
                    if 'spc_mask_vec' in self.opt.model.inputs else 0,

                    'text_bert_indices': text_bert_indices
                    if 'text_bert_indices' in self.opt.model.inputs else 0,

                    'aspect_bert_indices': aspect_bert_indices
                    if 'aspect_bert_indices' in self.opt.model.inputs else 0,

                    'text_raw_bert_indices': text_raw_bert_indices
                    if 'text_raw_bert_indices' in self.opt.model.inputs else 0,

                    'polarity': polarity,
                }

                x2ms_adapter.tensor_api.add(label_set, polarity)

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text)
                else:
                    raise e

        self.opt.polarities_dim = len(label_set)

        if self.opt.model_name in ['slide_lcf_bert', 'slide_lcfs_bert', 'ssw_t', 'ssw_s']:
            all_data = build_sentiment_window(all_data, self.tokenizer, self.opt.similarity_threshold)
            for data in all_data:

                cluster_ids = []
                for pad_idx in range(self.opt.max_seq_len):
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

        self.all_data = all_data
        return all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)
