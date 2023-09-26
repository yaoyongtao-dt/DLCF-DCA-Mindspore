# -*- coding: utf-8 -*-
# file: file_utils.py
# time: 2021/7/13 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import copy
import json
import os
import pickle
import urllib.request
from findfile import find_files, find_dir
from google_drive_downloader import GoogleDriveDownloader as gdd
from termcolor import colored

from pyabsa import __version__

# convert atepc_datasets in this repo for inferring_tutorials
from pyabsa.functional.dataset import DatasetItem
from pyabsa.utils.pyabsa_utils import save_args
import mindspore
import x2ms_adapter


def generate_inference_set_for_apc(dataset_path):
    if isinstance(dataset_path, DatasetItem):
        dataset_path = dataset_path.dataset_name
    train_datasets = find_files(os.getcwd(), ['dataset', 'train', 'apc', dataset_path], exclude_key='infer')
    test_datasets = find_files(os.getcwd(), ['dataset', 'test', 'apc', dataset_path], exclude_key='infer')
    for file in train_datasets + test_datasets:
        try:
            fin = open(file, 'r', newline='\n', encoding='utf-8')
            lines = fin.readlines()
            fin.close()
            path_to_save = file + '.inference'
            fout = open(path_to_save, 'w', encoding='utf-8', newline='\n', errors='ignore')

            for i in range(0, len(lines), 3):
                sample = lines[i].strip().replace('$T$', '[ASP]{}[ASP]'.format(lines[i + 1].strip()))
                fout.write(sample + ' !sent! ' + lines[i + 2].strip() + '\n')
            fout.close()
        except:
            print('Unprocessed file:', file)
        print('save in: {}'.format(path_to_save))
    print('process finished')


def is_similar(s1, s2):
    count = 0.0
    for token in x2ms_adapter.tensor_api.split(s1, ' '):
        if token in s2:
            count += 1
    if count / len(x2ms_adapter.tensor_api.split(s1, ' ')) >= 0.8 and count / len(x2ms_adapter.tensor_api.split(s2, ' ')) >= 0.8:
        return True
    else:
        return False


def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('$ t $', '$T$').strip()

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-999] * len(x2ms_adapter.tensor_api.split(text))
        tags = ['O'] * len(x2ms_adapter.tensor_api.split(text))
        samples = []
        for sample in same_samples:
            # print(sample)
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (x2ms_adapter.tensor_api.split(sample[0]).index('$T$'))
                asp_end = x2ms_adapter.tensor_api.split(sample[0]).index('$T$') + len(x2ms_adapter.tensor_api.split(sample[1]))
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = int(sample[2])
                    if i - x2ms_adapter.tensor_api.split(sample[0]).index('$T$') < 1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
                samples.append([text, tags, polarities_tmp])
            except:
                print('Ignore Error:', sample[0])

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        lines[i] = lines[i].replace('$T$', ' $T$ ').replace('  ', ' ')

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])

    return samples


def split_aspects(sentence):
    single_aspect_with_contex = []

    aspect_num = len(x2ms_adapter.tensor_api.split(sentence[1], "|"))
    aspects = x2ms_adapter.tensor_api.split(sentence[1], "|")
    polarity = x2ms_adapter.tensor_api.split(sentence[2], "|")
    pre_position = 0
    aspect_contex = sentence[0]
    for i in range(aspect_num):
        aspect_contex = aspect_contex.replace("$A$", aspects[i], 1)
        single_aspect_with_contex.append(
            (aspect_contex[pre_position:aspect_contex.find("$A$")], aspects[i], polarity[i]))
        pre_position = aspect_contex.find(aspects[i]) + len(aspects[i]) + 1

    return single_aspect_with_contex


def convert_atepc(fname):
    print('converting:', fname)
    dist_fname = fname + '.atepc'
    lines = []
    samples = assemble_aspects(fname)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarity = x2ms_adapter.tensor_api.split(sample[0])[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarity))
        lines.append('\n')

    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    fout = open(dist_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


# 将数据集中的aspect切割出来
def convert_apc_set_to_atepc_set(path):
    if isinstance(path, DatasetItem):
        path = path.dataset_name
    if not os.path.exists(path):
        files = find_files(os.getcwd(), [path, 'dataset', 'apc'], exclude_key='infer')
    else:
        files = find_files(path, '', exclude_key='infer')

    print('Find datasets files at {}:'.format(path))
    for f in files:
        print(f)
    for target_file in files:
        if not (target_file.endswith('.inference') or target_file.endswith('.atepc')):
            try:
                convert_atepc(target_file)
            except:
                print('failed to process"{}'.format(target_file))
        else:
            print('Ignore ', target_file)
    print('finished')


# 将数据集中的aspect切割出来
def refactor_chinese_dataset(fname, train_fname, test_fname):
    lines = []
    samples = assemble_aspects(fname)
    positive = 0
    negative = 0
    sum = 0
    # refactor testset
    for sample in samples[:int(len(samples) / 5)]:
        for token_index in range(len(sample[1])):
            token, label, polarty = x2ms_adapter.tensor_api.split(sample[0])[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(test_fname):
        os.remove(test_fname)
    fout = open(test_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()

    positive = 0
    negative = 0
    sum = 0
    # refactor trainset
    for sample in samples[int(len(samples) / 5):]:
        for token_index in range(len(sample[1])):
            tokens = x2ms_adapter.tensor_api.split(sample[0])
            token, label, polarty = x2ms_adapter.tensor_api.split(sample[0])[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum += 1
    print(train_fname + f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(train_fname):
        os.remove(train_fname)
    fout = open(train_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


def detect_error_in_dataset(dataset):
    f = open(dataset, 'r', encoding='utf8')
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        # print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
        if i + 3 < len(lines):
            if is_similar(lines[i], lines[i + 3]) and len(x2ms_adapter.tensor_api.split((lines[i] + " " + lines[i + 1]))) != len(
                    x2ms_adapter.tensor_api.split((lines[i + 3] + " " + lines[i + 4]))):
                print(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
                print(lines[i + 3].replace('$T$', lines[i + 4].replace('\n', '')))


def save_model(opt, model, tokenizer, save_path):
    if not opt.save_mode:
        return
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'core') else model  # Only save the model it-self

    if opt.save_mode == 1 or 'bert' not in opt.model_name:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            # torch.save(self.model.cpu().state_dict(), save_path + self.opt.model_name + '.state_dict')  # save the state dict
        x2ms_adapter.save(x2ms_adapter.state_dict(model), save_path + opt.model_name + '.state_dict')  # save the state dict
        with open(save_path + opt.model_name + '.config', mode='wb') as f:
            pickle.dump(opt, f)
        # pickle.dump(opt, open(save_path + opt.model_name + '.config', mode='wb'))
        with open(save_path + opt.model_name + '.tokenizer', mode='wb') as f:
            pickle.dump(tokenizer, f)
        # pickle.dump(tokenizer, open(save_path + opt.model_name + '.tokenizer', mode='wb'))
        save_args(opt, save_path + opt.model_name + '.args.txt')
    elif opt.save_mode == 2 or 'bert' not in opt.model_name:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(self.model.cpu().state_dict(), save_path + self.opt.model_name + '.state_dict')  # save the state dict
        x2ms_adapter.save(model, save_path + opt.model_name + '.model')  # save the state dict
        pickle.dump(opt, open(save_path + opt.model_name + '.config', mode='wb'))
        pickle.dump(tokenizer, open(save_path + opt.model_name + '.tokenizer', mode='wb'))
        save_args(opt, save_path + opt.model_name + '.args.txt')

    elif opt.save_mode == 3:
        # save the fine-tuned bert model
        model_output_dir = save_path + '-fine-tuned-bert'
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        output_model_file = os.path.join(model_output_dir, 'pytorch_model.bin')
        output_config_file = os.path.join(model_output_dir, 'bert_config.json')

        x2ms_adapter.save(x2ms_adapter.state_dict(model_to_save), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(model_output_dir)

    x2ms_adapter.to(model, opt.device)


def check_update_log():
    try:
        if os.path.exists('./release_note.json'):
            os.remove('./release_note.json')
        gdd.download_file_from_google_drive('1nOppewL8L1mGy9i6HQnJrEWrfaqQhC_2', './release-note.json')
        update_logs = json.load(open('release-note.json'))
        for v in update_logs:
            if v > __version__:
                print(colored('*' * 20 + ' Release Note of Version {} '.format(v) + '*' * 20, 'green'))
                for i, line in enumerate(update_logs[v]):
                    print('{}.\t{}'.format(i + 1, update_logs[v][line]))
    except Exception as e:
        print(colored('Fail to load release note: {}, you can check it on https://github.com/yangheng95/PyABSA/blob/release/release-note.json'.format(e), 'red'))


def check_dataset(dataset_path='./integrated_datasets', retry_count=3):  # retry_count is for unstable conn to GitHub
    try:
        with open(os.path.join(dataset_path, '__init__.py')) as f:
            local_version = x2ms_adapter.tensor_api.split(f.read(), '\'')[-2]
        # local_version = open(os.path.join(dataset_path, '__init__.py')).read().split('\'')[-2]

        if retry_count:
            def query_datasets():
                dataset_url = 'https://raw.githubusercontent.com/yangheng95/ABSADatasets/master/datasets/__init__.py'
                content = urllib.request.urlopen(dataset_url, timeout=int(5 / retry_count))
                version = x2ms_adapter.tensor_api.split(content.read().decode("utf-8"), '\'')[-2]
                return version

            try:
                result = query_datasets() > local_version
                if result:
                    print(colored('There is a new version of ABSADatasets, please remove the downloaded datasets to automatically download the new version.', 'green'))
            except Exception:
                retry_count -= 1
                check_dataset(retry_count=retry_count)
    except Exception as e:
        if find_dir('integrated_datasets'):
            print(colored('ABSADatasets version check failed, please check the latest datasets on GitHub manually.', 'red'))
