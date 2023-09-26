# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import tqdm

from pyabsa.core.apc.dataset_utils.apc_utils import configure_spacy_model
from pyabsa.core.atepc.dataset_utils.atepc_utils import prepare_input_for_atepc
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, SENTIMENT_PADDING
import x2ms_adapter


class InputExample(object):
    """A single training_tutorials/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, IOB_label=None, aspect_label=None, polarity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence core, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair core.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.IOB_label = IOB_label
        self.aspect_label = aspect_label
        self.polarity = polarity


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc,
                 input_mask,
                 segment_ids,
                 label_id,
                 polarity=None,
                 valid_ids=None,
                 label_mask=None,
                 tokens=None,
                 lcf_cdm_vec=None,
                 lcf_cdw_vec=None
                 ):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarity = polarity
        self.tokens = tokens
        self.lcf_cdm_vec = lcf_cdm_vec
        self.lcf_cdw_vec = lcf_cdw_vec


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = x2ms_adapter.tensor_api.split(line.strip(), ' ')
        if len(splits) != 3:
            print('warning! ignore detected error line(s) in input file:{}'.format(line))
            break
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1]))

    prepared_data = []
    for s, t, p in data:

        if len(s) > 0:
            # prepare the atepc dataset, refer to https://github.com/yangheng95/PyABSA/issues/78
            polarity_padding = [SENTIMENT_PADDING] * len(t)
            for p_idx in range(len(p) - 1):
                if (p[p_idx] != p[p_idx + 1] and p[p_idx] != SENTIMENT_PADDING and p[p_idx + 1] != SENTIMENT_PADDING) \
                        or (p[p_idx] != SENTIMENT_PADDING and p[p_idx + 1] == SENTIMENT_PADDING):
                    _p = p[:p_idx + 1] + polarity_padding[p_idx + 1:]
                    p = polarity_padding[:p_idx + 1] + p[p_idx + 1:]
                    prepared_data.append((s, t, _p))

    return prepared_data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = []
        for file in input_file:
            data += readfile(file)
        return data


class ATEPCProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.bos_token = tokenizer.bos_token if tokenizer.bos_token else '[CLS]'
        self.tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token else '[SEP]'

    def get_train_examples(self, data_dir, set_tag):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), set_tag)

    def get_test_examples(self, data_dir, set_tag):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), set_tag)

    def get_labels(self):
        return ["O", "B-ASP", "I-ASP", self.tokenizer.bos_token, self.tokenizer.eos_token]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, tag, polarity) in enumerate(lines):
            aspect = []
            aspect_tag = []
            aspect_polarity = SENTIMENT_PADDING
            for w, t, p in zip(sentence, tag, polarity):
                if p != SENTIMENT_PADDING:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity = p

            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, IOB_label=tag,
                                         aspect_label=aspect_tag, polarity=aspect_polarity))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer, opt=None):
    """Loads a data file into a list of `InputBatch`s."""

    configure_spacy_model(opt)

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []
    polarities_set = set()
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, postfix='convert examples to features')):
        text_spc_tokens = example.text_a[:]
        aspect_tokens = example.text_b[:]
        IOB_label = example.IOB_label
        aspect_label = example.aspect_label
        polarity = example.polarity
        if polarity != SENTIMENT_PADDING:  # bad case handle in Chinese atepc_datasets
            if polarity == -1:
                print(text_spc_tokens)
            x2ms_adapter.tensor_api.add(polarities_set, polarity)  # ignore samples without polarities
        tokens = []
        labels = []
        valid = []
        label_mask = []
        text_spc_tokens.extend([eos_token])
        text_spc_tokens.extend(aspect_tokens)
        enum_tokens = text_spc_tokens
        IOB_label.extend([eos_token])

        aspect = ' '.join(example.text_b)
        try:
            text_left, _, text_right = [s.strip() for s in ' '.join(example.text_a).partition(aspect)]
        except:
            text_left = ' '.join(example.text_a)
            text_right = ''
            aspect = ''
        prepared_inputs = prepare_input_for_atepc(opt, tokenizer, text_left, text_right, aspect)
        lcf_cdm_vec = prepared_inputs['lcf_cdm_vec']
        lcf_cdw_vec = prepared_inputs['lcf_cdw_vec']

        IOB_label.extend(aspect_label)
        label_lists = IOB_label
        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_lists[i]
            for m in range(len(token)):
                if m == 0:
                    label_mask.append(1)
                    labels.append(label_1)
                    valid.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]
            labels = labels[0:(max_seq_len - 2)]
            valid = valid[0:(max_seq_len - 2)]
            label_mask = label_mask[0:(max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append(bos_token)
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map[bos_token])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append(eos_token)
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map[eos_token])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        while len(input_ids_spc) < max_seq_len:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            label_mask.append(0)
            while len(valid) < max_seq_len:
                valid.append(1)
        while len(label_ids) < max_seq_len:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids_spc) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(valid) == max_seq_len
        assert len(label_mask) == max_seq_len

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarity=polarity,
                          valid_ids=valid,
                          label_mask=label_mask,
                          tokens=example.text_a,
                          lcf_cdm_vec=lcf_cdm_vec,
                          lcf_cdw_vec=lcf_cdw_vec)
        )
    check_and_fix_labels(polarities_set, 'polarity', features, opt)
    opt.polarities_dim = len(polarities_set)

    return features
