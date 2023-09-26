# -*- coding: utf-8 -*-
# file: apc_trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import mindspore
import mindspore.context as context
import x2ms_adapter
from x2ms_adapter.context import x2ms_context
from x2ms_adapter.optimizers import optim_register
from x2ms_adapter.exception import TrainBreakException, TrainContinueException, TrainReturnException
import mindspore
import x2ms_adapter
import x2ms_adapter.datasets as x2ms_datasets
import x2ms_adapter.loss as loss_wrapper
import x2ms_adapter.nn_cell
import x2ms_adapter.nn_init
import x2ms_adapter.numpy as x2ms_np
import x2ms_adapter.util_api as util_api

if not x2ms_context.is_context_init:
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    x2ms_context.is_context_init = True
import math
import os
import random
import shutil
import time

import numpy
from findfile import find_file
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
# , BertModel
from mindnlp.models import BertModel
# from mindnlp.transforms.tokenizers import BertTokenizer

from pyabsa.functional.dataset import ABSADatasetList
from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, optimizers, load_checkpoint, retry

from ..models import BERTBaselineAPCModelList, GloVeAPCModelList, APCModelList
from ..classic.__bert__.dataset_utils.data_utils_for_training import (Tokenizer4Pretraining,
                                                                      BERTBaselineABSADataset)
from ..classic.__glove__.dataset_utils.data_utils_for_training import (build_tokenizer,
                                                                       build_embedding_matrix,
                                                                       GloVeABSADataset)
from ..dataset_utils.data_utils_for_training import ABSADataset


class Instructor:
    def __init__(self, opt, logger):
        self.logger = logger
        self.opt = opt

        # init BERT-based model and dataset_manager
        if hasattr(APCModelList, opt.model.__name__):
            self.tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case=True)
            # 这里改过
            self.train_set = ABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)
            if self.opt.dataset_file['test']:
                self.test_set = ABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
                self.test_dataloader = x2ms_datasets.DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
            else:
                self.test_set = None

            self.bert = BertModel.from_pretrained(self.opt.pretrained_bert)
            # init the model behind the construction of apc_datasets in case of updating polarities_dim
            self.model = x2ms_adapter.to(self.opt.model(self.bert, self.opt), self.opt.device)

        elif hasattr(BERTBaselineAPCModelList, opt.model.__name__):
            self.tokenizer = Tokenizer4Pretraining(self.opt.max_seq_len, self.opt.pretrained_bert)

            self.train_set = BERTBaselineABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)
            if self.opt.dataset_file['test']:
                self.test_set = BERTBaselineABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
                self.test_dataloader = x2ms_datasets.DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
            else:
                self.test_set = None
            # 改过这里
            self.bert = BertModel.from_pretrained(self.opt.pretrained_bert)
            # init the model behind the construction of apc_datasets in case of updating polarities_dim
            self.model = x2ms_adapter.to(self.opt.model(self.bert, self.opt), self.opt.device)

        elif hasattr(GloVeAPCModelList, opt.model.__name__):
            # init GloVe-based model and dataset_manager

            if hasattr(ABSADatasetList, opt.dataset_name):
                opt.dataset_name = os.path.join(os.getcwd(), opt.dataset_name)
                if not os.path.exists(os.path.join(os.getcwd(), opt.dataset_name)):
                    os.mkdir(os.path.join(os.getcwd(), opt.dataset_name))

            self.tokenizer = build_tokenizer(
                dataset_list=opt.dataset_file,
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(os.path.basename(opt.dataset_name)),
                opt=self.opt
            )
            self.embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), os.path.basename(opt.dataset_name)),
                opt=self.opt
            )

            self.train_set = GloVeABSADataset(self.opt.dataset_file['train'], self.tokenizer, self.opt)
            if self.opt.dataset_file['test']:
                self.test_set = GloVeABSADataset(self.opt.dataset_file['test'], self.tokenizer, self.opt)
                self.test_dataloader = x2ms_datasets.DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)

            else:
                self.test_set = None

            self.model = x2ms_adapter.to(opt.model(self.embedding_matrix, opt), opt.device)

        if self.opt.device.type == 'cuda':
            self.logger.info(
                "cuda memory allocated:{}".format(x2ms_adapter.memory_allocated(device=self.opt.device.index)))

        print_args(self.opt, self.logger)

        initializers = {
            'xavier_uniform_': x2ms_adapter.nn_init.xavier_uniform_,
            'xavier_normal_': x2ms_adapter.nn_init.xavier_normal_,
            'orthogonal_': x2ms_adapter.nn_init.orthogonal_,
        }
        self.initializer = initializers[self.opt.initializer]

        self.optimizer = optimizers[self.opt.optimizer](
            x2ms_adapter.parameters(self.model),
            lr=self.opt.learning_rate,
            weight_decay=self.opt.l2reg
        )
        self.train_dataloaders = []
        self.val_dataloaders = []

        if os.path.exists('init_state_dict.tmp'):
            os.remove('init_state_dict.tmp')
        if self.opt.cross_validate_fold > 0:
            x2ms_adapter.save(x2ms_adapter.state_dict(self.model), 'init_state_dict.tmp')

    def _reset_params(self):
        for child in x2ms_adapter.nn_cell.children(self.model):
            if type(child) != BertModel:  # skip bert params
                for p in x2ms_adapter.parameters(child):
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.initializer(p)
                        else:
                            stdv = 1. / x2ms_adapter.tensor_api.sqrt(math, p.shape[0])
                            x2ms_adapter.nn_init.uniform_(p, a=-stdv, b=stdv)

    def reload_model(self):
        x2ms_adapter.load_state_dict(self.model, x2ms_adapter.load('./init_state_dict.bin'))
        _params = filter(lambda p: p.requires_grad, x2ms_adapter.parameters(self.model))
        self.optimizer = optimizers[self.opt.optimizer](_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    def prepare_dataloader(self, train_set):
        if self.opt.cross_validate_fold < 1:
            self.train_dataloaders.append(x2ms_datasets.DataLoader(dataset=train_set,
                                                     batch_size=self.opt.batch_size,
                                                     shuffle=True,
                                                     pin_memory=True))

        else:
            split_dataset = train_set
            len_per_fold = len(split_dataset) // self.opt.cross_validate_fold
            folds = x2ms_datasets.random_split(split_dataset, tuple([len_per_fold] * (self.opt.cross_validate_fold - 1) + [
                len(split_dataset) - len_per_fold * (self.opt.cross_validate_fold - 1)]))

            for f_idx in range(self.opt.cross_validate_fold):
                train_set = util_api.ConcatDataset([x for i, x in enumerate(folds) if i != f_idx])
                val_set = folds[f_idx]
                self.train_dataloaders.append(
                    x2ms_datasets.DataLoader(dataset=train_set, batch_size=self.opt.batch_size, shuffle=True))
                self.val_dataloaders.append(
                    x2ms_datasets.DataLoader(dataset=val_set, batch_size=self.opt.batch_size, shuffle=True))

    def _train(self, criterion):
        self.prepare_dataloader(self.train_set)
        if self.val_dataloaders:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)

    def _train_and_evaluate(self, criterion):

        sum_loss = 0
        sum_acc = 0
        sum_f1 = 0

        global_step = 0
        max_fold_acc = 0
        max_fold_f1 = 0
        save_path = ''
        self.opt.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0, 'max_ate_test_f1': 0}

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in x2ms_adapter.parameters(self.model):
            mulValue = x2ms_adapter.tensor_api.prod(numpy, x2ms_adapter.tensor_api.x2ms_size(param))  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        self.logger.info("***** Running training for Aspect Polarity Classification *****")
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Total params = %d, Trainable params = %d, Non-trainable params = %d", Total_params,
                         Trainable_params, NonTrainable_params)
        self.logger.info("Batch size = %d", self.opt.batch_size)
        self.logger.info("Num steps = %d", len(self.train_dataloaders[0]) // self.opt.batch_size * self.opt.num_epoch)

        for epoch in range(self.opt.num_epoch):
            iterator = tqdm(self.train_dataloaders[0])
            class WithLossCell(mindspore.nn.Cell):
                def __init__(self, train_obj=None):
                    super(WithLossCell, self).__init__(auto_prefix=False)
                    self._input = None
                    self._output = None
                    self.train_obj = train_obj
                    self.amp_model = x2ms_context.amp_model

                def construct(self):
                    nonlocal global_step
                    nonlocal sum_loss
                    loss, sen_logits = None, None
                    i_batch, sample_batched = self._input
                    global_step += 1
                    # switch model to training_tutorials mode, clear gradient accumulators
                    x2ms_adapter.x2ms_train(self.train_obj.model)
                    x2ms_adapter.nn_cell.zero_grad(self.train_obj.optimizer)
                    inputs = [x2ms_adapter.to(sample_batched[col], self.train_obj.opt.device) for col in self.train_obj.opt.inputs_cols]
                    outputs = self.train_obj.model(inputs)
                    targets = x2ms_adapter.to(sample_batched['polarity'], self.train_obj.opt.device)

                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        sen_logits = outputs
                        loss = criterion(sen_logits, targets)

                    sum_loss += x2ms_adapter.tensor_api.item(loss)
                    self._output = (inputs, loss, outputs, sen_logits, targets)
                    return loss

                @property
                def output(self):
                    return self._output
            wrapped_model = WithLossCell(self)
            wrapped_model = x2ms_adapter.train_one_step_cell(wrapped_model, optim_register.get_instance())
            for i_batch, sample_batched in enumerate(iterator):
                try:
                    wrapped_model.network._input = (i_batch, sample_batched)
                    wrapped_model()
                except TrainBreakException:
                    break
                except TrainContinueException:
                    continue
                except TrainReturnException:
                    return

                inputs, loss, outputs, sen_logits, targets = wrapped_model.network.output
                self.optimizer.step()

                # evaluate if test set is available
                if self.opt.dataset_file['test'] and global_step % self.opt.log_step == 0:
                    if epoch >= self.opt.evaluate_begin:

                        test_acc, f1 = self._evaluate_acc_f1(self.test_dataloader)

                        self.opt.metrics_of_this_checkpoint['acc'] = test_acc
                        self.opt.metrics_of_this_checkpoint['f1'] = f1

                        sum_acc += test_acc
                        sum_f1 += f1
                        if test_acc > max_fold_acc:

                            max_fold_acc = test_acc
                            if self.opt.model_path_to_save:
                                if not os.path.exists(self.opt.model_path_to_save):
                                    os.mkdir(self.opt.model_path_to_save)
                                if save_path:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-optimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-optimal trained model:', save_path)
                                        pass
                                save_path = '{0}/{1}_acc_{2}_f1_{3}/'.format(self.opt.model_path_to_save,
                                                                             self.opt.model_name,
                                                                             round(test_acc * 100, 2),
                                                                             round(f1 * 100, 2)
                                                                             )

                                if test_acc > self.opt.max_test_metrics['max_apc_test_acc']:
                                    self.opt.max_test_metrics['max_apc_test_acc'] = test_acc
                                if f1 > self.opt.max_test_metrics['max_apc_test_f1']:
                                    self.opt.max_test_metrics['max_apc_test_f1'] = f1

                                save_model(self.opt, self.model, self.tokenizer, save_path)
                        if f1 > max_fold_f1:
                            max_fold_f1 = f1
                        postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                   ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                        x2ms_adapter.tensor_api.item(loss),
                                                                        test_acc * 100,
                                                                        max_fold_acc * 100,
                                                                        f1 * 100,
                                                                        max_fold_f1 * 100))
                    else:
                        postfix = 'Epoch:{} | No evaluation until epoch:{}'.format(epoch, self.opt.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()

        self.logger.info('-------------------------- Training Summary --------------------------')
        self.logger.info('Acc: {:.8f} F1: {:.8f} Accumulated Loss: {:.8f}'.format(
            max_fold_acc * 100,
            max_fold_f1 * 100,
            sum_loss)
        )
        self.logger.info('-------------------------- Training Summary --------------------------')
        if os.path.exists('./init_state_dict.bin'):
            self.reload_model()

        print('Training finished, we hope you can share your checkpoint with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        if save_path:
            return save_path
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path = '{0}/{1}/'.format(self.opt.model_path_to_save,
                                              self.opt.model_name
                                              )
                save_model(self.opt, self.model, self.tokenizer, save_path)
            return self.model, self.opt, self.tokenizer, sum_acc, sum_f1

    def _k_fold_train_and_evaluate(self, criterion):
        sum_loss = 0
        sum_acc = 0
        sum_f1 = 0

        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ''
        max_fold_acc_k_fold = 0

        self.opt.metrics_of_this_checkpoint = {'acc': 0, 'f1': 0}
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0, 'max_ate_test_f1': 0}

        for f, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloaders, self.val_dataloaders)):
            self.logger.info("***** Running training for Aspect Polarity Classification *****")
            self.logger.info("Training set examples = %d", len(self.train_set))
            if self.test_set:
                self.logger.info("Test set examples = %d", len(self.test_set))
            self.logger.info("Batch size = %d", self.opt.batch_size)
            self.logger.info("Num steps = %d", len(train_dataloader) // self.opt.batch_size * self.opt.num_epoch)
            if len(self.train_dataloaders) > 1:
                self.logger.info('No. {} training in {} folds...'.format(f + 1, self.opt.cross_validate_fold))
            global_step = 0
            max_fold_acc = 0
            max_fold_f1 = 0
            save_path = ''
            for epoch in range(self.opt.num_epoch):
                iterator = tqdm(train_dataloader)
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to training_tutorials mode, clear gradient accumulators
                    x2ms_adapter.x2ms_train(self.model)
                    x2ms_adapter.nn_cell.zero_grad(self.optimizer)
                    inputs = [x2ms_adapter.to(sample_batched[col], self.opt.device) for col in self.opt.inputs_cols]
                    outputs = self.model(inputs)
                    targets = x2ms_adapter.to(sample_batched['polarity'], self.opt.device)

                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        sen_logits = outputs
                        loss = criterion(sen_logits, targets)

                    sum_loss += x2ms_adapter.tensor_api.item(loss)
                    loss.backward()
                    self.optimizer.step()

                    # evaluate if test set is available
                    if self.opt.dataset_file['test'] and global_step % self.opt.log_step == 0:
                        if epoch >= self.opt.evaluate_begin:

                            test_acc, f1 = self._evaluate_acc_f1(val_dataloader)

                            self.opt.metrics_of_this_checkpoint['acc'] = test_acc
                            self.opt.metrics_of_this_checkpoint['f1'] = f1

                            sum_acc += test_acc
                            sum_f1 += f1
                            if test_acc > max_fold_acc:

                                max_fold_acc = test_acc
                                if self.opt.model_path_to_save:
                                    if not os.path.exists(self.opt.model_path_to_save):
                                        os.mkdir(self.opt.model_path_to_save)
                                    if save_path:
                                        try:
                                            shutil.rmtree(save_path)
                                            # logger.info('Remove sub-optimal trained model:', save_path)
                                        except:
                                            # logger.info('Can not remove sub-optimal trained model:', save_path)
                                            pass
                                    save_path = '{0}/{1}_acc_{2}_f1_{3}/'.format(self.opt.model_path_to_save,
                                                                                 self.opt.model_name,
                                                                                 round(test_acc * 100, 2),
                                                                                 round(f1 * 100, 2)
                                                                                 )

                                    if test_acc > self.opt.max_test_metrics['max_apc_test_acc']:
                                        self.opt.max_test_metrics['max_apc_test_acc'] = test_acc
                                    if f1 > self.opt.max_test_metrics['max_apc_test_f1']:
                                        self.opt.max_test_metrics['max_apc_test_f1'] = f1

                                    save_model(self.opt, self.model, self.tokenizer, save_path)
                            if f1 > max_fold_f1:
                                max_fold_f1 = f1
                            postfix = ('Epoch:{} | Loss:{:.4f} | Test Acc:{:.2f}(max:{:.2f}) |'
                                       ' Test F1:{:.2f}(max:{:.2f})'.format(epoch,
                                                                            x2ms_adapter.tensor_api.item(loss),
                                                                            test_acc * 100,
                                                                            max_fold_acc * 100,
                                                                            f1 * 100,
                                                                            max_fold_f1 * 100))
                        else:
                            postfix = 'Epoch:{} | No evaluation until epoch:{}'.format(epoch, self.opt.evaluate_begin)

                        iterator.postfix = postfix
                        iterator.refresh()
            x2ms_adapter.load_state_dict(self.model, x2ms_adapter.load(find_file(None, 'state_dict')))
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)
            self.logger.info('-------------------------- Training Summary --------------------------')
            self.logger.info('Acc: {:.8f} F1: {:.8f} Accumulated Loss: {:.8f}'.format(
                max_fold_acc * 100,
                max_fold_f1 * 100,
                sum_loss)
            )
            self.logger.info('-------------------------- Training Summary --------------------------')
            if os.path.exists('./init_state_dict.bin'):
                self.reload_model()

        mean_test_acc = x2ms_np.mean(fold_test_acc)
        mean_test_f1 = x2ms_np.mean(fold_test_f1)

        if self.opt.cross_validate_fold > 0:
            self.logger.info('-------------------------- Training Summary --------------------------')
            self.logger.info('{}-fold Avg Acc: {:.8f} Avg F1: {:.8f} Accumulated Loss: {:.8f}'.format(
                self.opt.cross_validate_fold,
                mean_test_acc * 100,
                mean_test_f1 * 100,
                sum_loss)
            )
            self.logger.info('-------------------------- Training Summary --------------------------')

        print('Training finished, we hope you can share your checkpoint with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        if os.path.exists('./init_state_dict.bin'):
            self.reload_model()
            os.remove('./init_state_dict.bin')
        if save_path_k_fold:
            return save_path_k_fold
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path_k_fold = '{0}/{1}/'.format(self.opt.model_path_to_save,
                                                     self.opt.model_name
                                                     )
                save_model(self.opt, self.model, self.tokenizer, save_path_k_fold)
            return self.model, self.opt, self.tokenizer, sum_acc, sum_f1

    def _evaluate_acc_f1(self, test_dataloader):
        # switch model to evaluation mode
        x2ms_adapter.x2ms_eval(self.model)
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        for t_batch, t_sample_batched in enumerate(test_dataloader):

            t_inputs = [x2ms_adapter.to(t_sample_batched[col], self.opt.device) for col in self.opt.inputs_cols]
            t_targets = x2ms_adapter.to(t_sample_batched['polarity'], self.opt.device)

            t_outputs = self.model(t_inputs)

            if isinstance(t_outputs, dict):
                sen_outputs = t_outputs['logits']
            else:
                sen_outputs = t_outputs

            n_test_correct += x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.x2ms_sum((x2ms_adapter.argmax(sen_outputs, -1) == t_targets)))
            n_test_total += len(sen_outputs)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = sen_outputs
            else:
                t_targets_all = x2ms_adapter.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = x2ms_adapter.cat((t_outputs_all, sen_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = x2ms_np.sklearn_metrics_f1_score(t_targets_all, x2ms_adapter.argmax(t_outputs_all, -1),
                              labels=list(range(self.opt.polarities_dim)), average='macro')
        return test_acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = loss_wrapper.CrossEntropyLoss()
        self._reset_params()
        return self._train(criterion)


# @retry
def train4apc(opt, from_checkpoint_path, logger):
    if not isinstance(opt.seed, int):
        opt.logger.info('Please do not use multiple random seeds without evaluating.')
        opt.seed = list(opt.seed)[0]
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    mindspore.set_seed(opt.seed)
    mindspore.set_seed(opt.seed)

    if hasattr(APCModelList, opt.model.__name__):
        opt.inputs_cols = opt.model.inputs

    elif hasattr(BERTBaselineAPCModelList, opt.model.__name__):
        opt.inputs_cols = opt.model.inputs

    elif hasattr(GloVeAPCModelList, opt.model.__name__):
        opt.inputs_cols = opt.model.inputs

    opt.device = x2ms_adapter.Device(opt.device)

    # in case of handling ConnectionError exception
    trainer = Instructor(opt, logger)
    load_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
