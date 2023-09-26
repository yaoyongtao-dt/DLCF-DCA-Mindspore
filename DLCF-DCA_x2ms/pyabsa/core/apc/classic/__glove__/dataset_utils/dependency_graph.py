import os.path
import pickle

import numpy as np
import spacy
import tqdm
from spacy.tokens import Doc
import x2ms_adapter
import x2ms_adapter.nn_cell


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = x2ms_adapter.tensor_api.split(text)
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = x2ms_adapter.tensor_api.split(text)
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def prepare_dependency_graph(dataset_list, graph_path, max_seq_len):
    if 'train' in dataset_list[0].lower():
        append_name = 'train_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    elif 'test' in dataset_list[0].lower():
        append_name = 'test_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    elif 'val' in dataset_list[0].lower():
        append_name = 'val_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    else:
        append_name = 'unrecognized_set_{}x{}.graph'.format(max_seq_len, max_seq_len)

    graph_path = os.path.join(graph_path, append_name)

    if os.path.isfile(graph_path):
        return graph_path

    idx2graph = {}
    if os.path.isdir(graph_path):
        fout = open(os.path.join(graph_path, append_name), 'wb')
        graph_path = os.path.join(graph_path, append_name)
    elif os.path.isfile(graph_path):
        return graph_path
    else:
        fout = open(graph_path, 'wb')

    for filename in dataset_list:
        try:
            print('parsing dependency matrix:', filename)
            fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()

            for i in tqdm.tqdm(range(0, len(lines), 3), postfix='Construct graph for {}'.format(filename)):
                text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].strip()
                adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
                idx2graph[i] = adj_matrix

        except Exception as e:
            print(e)
            print('unprocessed:', filename)
    pickle.dump(idx2graph, fout)
    fout.close()
    return graph_path
