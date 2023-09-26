# -*- coding: utf-8 -*-
# file: test_train_atepc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import mindspore
import mindspore.context as context
import x2ms_adapter
from x2ms_adapter.context import x2ms_context
from x2ms_adapter.optimizers import optim_register
from x2ms_adapter.exception import TrainBreakException, TrainContinueException, TrainReturnException
from x2ms_adapter.optimizers import optim_register
import mindspore
import x2ms_adapter
import x2ms_adapter.datasets as x2ms_datasets
import x2ms_adapter.nn_cell
import x2ms_adapter.nn_functional
import x2ms_adapter.numpy as x2ms_np

if not x2ms_context.is_context_init:
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    x2ms_context.is_context_init = True
import os
import random
import time

import numpy as np
import tqdm
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
# from transformers import AutoTokenizer, AutoModel
from mindnlp.models import BertModel
from mindnlp.transforms.tokenizers import BertTokenizer

from pyabsa.utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import print_args, load_checkpoint, retry
from ..dataset_utils.data_utils_for_training import ATEPCProcessor, convert_examples_to_features


class Instructor:

    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        if opt.use_bert_spc:
            self.logger.info('Warning: The use_bert_spc is disabled for extracting aspect,'
                             ' reset use_bert_spc=False and go on... ')
            opt.use_bert_spc = False
        import warnings
        warnings.filterwarnings('ignore')
        if self.opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.opt.gradient_accumulation_steps))

        self.opt.batch_size = self.opt.batch_size // self.opt.gradient_accumulation_steps

        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        mindspore.set_seed(self.opt.seed)
        mindspore.set_seed(self.opt.seed)

        if self.opt.model_path_to_save and not os.path.exists(self.opt.model_path_to_save):
            os.makedirs(self.opt.model_path_to_save)

        self.optimizers = {
            'adam': optim_register.adam,
            'adamw': optim_register.adamw
        }

        self.tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_bert, do_lower_case=True)
        bert_base_model = BertModel.from_pretrained(self.opt.pretrained_bert)
        processor = ATEPCProcessor(self.tokenizer)
        self.label_list = processor.get_labels()
        self.opt.num_labels = len(self.label_list) + 1

        bert_base_model.config.num_labels = self.opt.num_labels

        self.train_examples = processor.get_train_examples(self.opt.dataset_file['train'], 'train')
        self.num_train_optimization_steps = int(
            len(self.train_examples) / self.opt.batch_size / self.opt.gradient_accumulation_steps) * self.opt.num_epoch
        train_features = convert_examples_to_features(self.train_examples, self.label_list, self.opt.max_seq_len,
                                                      self.tokenizer, self.opt)
        all_spc_input_ids = x2ms_adapter.x2ms_tensor([f.input_ids_spc for f in train_features], dtype=mindspore.int64)
        all_input_mask = x2ms_adapter.x2ms_tensor([f.input_mask for f in train_features], dtype=mindspore.int64)
        all_segment_ids = x2ms_adapter.x2ms_tensor([f.segment_ids for f in train_features], dtype=mindspore.int64)
        all_label_ids = x2ms_adapter.x2ms_tensor([f.label_id for f in train_features], dtype=mindspore.int64)
        all_valid_ids = x2ms_adapter.x2ms_tensor([f.valid_ids for f in train_features], dtype=mindspore.int64)
        all_lmask_ids = x2ms_adapter.x2ms_tensor([f.label_mask for f in train_features], dtype=mindspore.int64)
        all_polarities = x2ms_adapter.x2ms_tensor([f.polarity for f in train_features], dtype=mindspore.int64)
        lcf_cdm_vec = x2ms_adapter.x2ms_tensor([f.lcf_cdm_vec for f in train_features], dtype=mindspore.float32)
        lcf_cdw_vec = x2ms_adapter.x2ms_tensor([f.lcf_cdw_vec for f in train_features], dtype=mindspore.float32)

        train_data = x2ms_datasets.TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids,
                                   all_polarities, all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)

        train_sampler = x2ms_datasets.SequentialSampler(train_data)
        self.train_dataloader = x2ms_datasets.DataLoader(train_data, sampler=train_sampler, batch_size=self.opt.batch_size)

        if 'test' in self.opt.dataset_file:
            eval_examples = processor.get_test_examples(self.opt.dataset_file['test'], 'test')
            eval_features = convert_examples_to_features(eval_examples, self.label_list, self.opt.max_seq_len,
                                                         self.tokenizer, self.opt)
            all_spc_input_ids = x2ms_adapter.x2ms_tensor([f.input_ids_spc for f in eval_features], dtype=mindspore.int64)
            all_input_mask = x2ms_adapter.x2ms_tensor([f.input_mask for f in eval_features], dtype=mindspore.int64)
            all_segment_ids = x2ms_adapter.x2ms_tensor([f.segment_ids for f in eval_features], dtype=mindspore.int64)
            all_label_ids = x2ms_adapter.x2ms_tensor([f.label_id for f in eval_features], dtype=mindspore.int64)
            all_polarities = x2ms_adapter.x2ms_tensor([f.polarity for f in eval_features], dtype=mindspore.int64)
            all_valid_ids = x2ms_adapter.x2ms_tensor([f.valid_ids for f in eval_features], dtype=mindspore.int64)
            all_lmask_ids = x2ms_adapter.x2ms_tensor([f.label_mask for f in eval_features], dtype=mindspore.int64)
            lcf_cdm_vec = x2ms_adapter.x2ms_tensor([f.lcf_cdm_vec for f in eval_features], dtype=mindspore.float32)
            lcf_cdw_vec = x2ms_adapter.x2ms_tensor([f.lcf_cdw_vec for f in eval_features], dtype=mindspore.float32)
            eval_data = x2ms_datasets.TensorDataset(all_spc_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_polarities,
                                      all_valid_ids, all_lmask_ids, lcf_cdm_vec, lcf_cdw_vec)
            # all_tokens = [f.tokens for f in eval_features]

            eval_sampler = x2ms_datasets.RandomSampler(eval_data)
            self.eval_dataloader = x2ms_datasets.DataLoader(eval_data, sampler=eval_sampler, batch_size=self.opt.batch_size)

        # init the model behind the convert_examples_to_features function in case of updating polarities_dim

        self.model = self.opt.model(bert_base_model, opt=self.opt)
        x2ms_adapter.to(self.model, self.opt.device)
        param_optimizer = list(x2ms_adapter.named_parameters(self.model))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg}
        ]
        if isinstance(self.opt.optimizer, str):
            self.optimizer = self.optimizers[self.opt.optimizer](self.optimizer_grouped_parameters,
                                                                 lr=self.opt.learning_rate,
                                                                 weight_decay=self.opt.l2reg)
        print_args(self.opt, self.logger)

    def run(self):

        self.logger.info("***** Running training for Aspect Term Extraction *****")
        self.logger.info("  Num examples = %d", len(self.train_examples))
        self.logger.info("  Batch size = %d", self.opt.batch_size)
        self.logger.info("  Num steps = %d", self.num_train_optimization_steps)
        sum_loss = 0
        sum_apc_test_acc = 0
        sum_apc_test_f1 = 0
        sum_ate_test_f1 = 0
        self.opt.max_test_metrics = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0, 'max_ate_test_f1': 0}
        self.opt.metrics_of_this_checkpoint = {'apc_acc': 0, 'apc_f1': 0, 'ate_f1': 0}
        global_step = 0
        save_path = ''
        for epoch in range(int(self.opt.num_epoch)):
            nb_tr_examples, nb_tr_steps = 0, 0
            iterator = tqdm.tqdm(self.train_dataloader)
            class WithLossCell(mindspore.nn.Cell):
                def __init__(self, train_obj=None):
                    super(WithLossCell, self).__init__(auto_prefix=False)
                    self._input = None
                    self._output = None
                    self.train_obj = train_obj
                    self.amp_model = x2ms_context.amp_model

                def construct(self):
                    nonlocal sum_loss
                    
                    step, batch = self._input
                    x2ms_adapter.x2ms_train(self.train_obj.model)
                    batch = tuple(x2ms_adapter.to(t, self.train_obj.opt.device) for t in batch)
                    input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
                valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch
                    loss_ate, loss_apc = self.train_obj.model(input_ids_spc,
                                                    token_type_ids=segment_ids,
                                                    attention_mask=input_mask,
                                                    labels=label_ids,
                                                    polarity=polarity,
                                                    valid_ids=valid_ids,
                                                    attention_mask_label=l_mask,
                                                    lcf_cdm_vec=lcf_cdm_vec,
                                                    lcf_cdw_vec=lcf_cdw_vec
                                                    )
                    # loss_ate = loss_ate.item() / (loss_ate.item() + loss_apc.item()) * loss_ate
                    # loss_apc = loss_apc.item() / (loss_ate.item() + loss_apc.item()) * loss_apc
                    loss = 3 * loss_ate + loss_apc
                    sum_loss += x2ms_adapter.tensor_api.item(loss)
                    self._output = (batch, input_ids_spc, input_mask, l_mask, label_ids, lcf_cdm_vec, lcf_cdw_vec, loss, loss_apc, loss_ate, polarity, segment_ids, valid_ids)
                    return loss

                @property
                def output(self):
                    return self._output
            wrapped_model = WithLossCell(self)
            wrapped_model = x2ms_adapter.train_one_step_cell(wrapped_model, optim_register.get_instance())
            for step, batch in enumerate(iterator):
                try:
                    wrapped_model.network._input = (step, batch)
                    wrapped_model()
                except TrainBreakException:
                    break
                except TrainContinueException:
                    continue
                except TrainReturnException:
                    return
                    
                batch, input_ids_spc, input_mask, l_mask, label_ids, lcf_cdm_vec, lcf_cdw_vec, loss, loss_apc, loss_ate, polarity, segment_ids, valid_ids = wrapped_model.network.output
                nb_tr_examples += x2ms_adapter.tensor_api.x2ms_size(input_ids_spc, 0)
                nb_tr_steps += 1
                self.optimizer.step()
                x2ms_adapter.nn_cell.zero_grad(self.optimizer)
                global_step += 1
                global_step += 1
                if 'test' in self.opt.dataset_file and global_step % self.opt.log_step == 0:
                    if epoch >= self.opt.evaluate_begin:
                        apc_result, ate_result = self.evaluate(
                            eval_ATE=not (self.opt.model_name == 'lcf_atepc' and self.opt.use_bert_spc))
                        sum_apc_test_acc += apc_result['apc_test_acc']
                        sum_apc_test_f1 += apc_result['apc_test_f1']
                        sum_ate_test_f1 += ate_result
                        self.opt.metrics_of_this_checkpoint['apc_acc'] = apc_result['apc_test_acc']
                        self.opt.metrics_of_this_checkpoint['apc_f1'] = apc_result['apc_test_f1']
                        self.opt.metrics_of_this_checkpoint['ate_f1'] = ate_result

                        if apc_result['apc_test_acc'] > self.opt.max_test_metrics['max_apc_test_acc'] or \
                                apc_result['apc_test_f1'] > self.opt.max_test_metrics['max_apc_test_f1'] or \
                                ate_result > self.opt.max_test_metrics['max_ate_test_f1']:

                            if apc_result['apc_test_acc'] > self.opt.max_test_metrics['max_apc_test_acc']:
                                self.opt.max_test_metrics['max_apc_test_acc'] = apc_result['apc_test_acc']
                            if apc_result['apc_test_f1'] > self.opt.max_test_metrics['max_apc_test_f1']:
                                self.opt.max_test_metrics['max_apc_test_f1'] = apc_result['apc_test_f1']
                            if ate_result > self.opt.max_test_metrics['max_ate_test_f1']:
                                self.opt.max_test_metrics['max_ate_test_f1'] = ate_result

                            if self.opt.model_path_to_save:
                                # if save_path:
                                #     try:
                                #         shutil.rmtree(save_path)
                                #         # self.logger.info('Remove sub-self.optimal trained model:', save_path)
                                #     except:
                                #         self.logger.info('Can not remove sub-self.optimal trained model:', save_path)

                                save_path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}/'.format(
                                    self.opt.model_path_to_save,
                                    self.opt.model_name,
                                    self.opt.lcf,
                                    round(apc_result['apc_test_acc'], 2),
                                    round(apc_result['apc_test_f1'], 2),
                                    round(ate_result, 2)
                                )

                                save_model(self.opt, self.model, self.tokenizer, save_path)

                        current_apc_test_acc = apc_result['apc_test_acc']
                        current_apc_test_f1 = apc_result['apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)

                        postfix = 'Epoch:{} | '.format(epoch)

                        postfix += 'loss_apc:{:.4f} | loss_ate:{:.4f} |'.format(x2ms_adapter.tensor_api.item(loss_apc), x2ms_adapter.tensor_api.item(loss_ate))

                        postfix += ' APC_ACC: {}(max:{}) | APC_F1: {}(max:{}) | '.format(current_apc_test_acc,
                                                                                         self.opt.max_test_metrics[
                                                                                             'max_apc_test_acc'],
                                                                                         current_apc_test_f1,
                                                                                         self.opt.max_test_metrics[
                                                                                             'max_apc_test_f1']
                                                                                         )
                        if self.opt.model_name == 'lcf_atepc' and self.opt.use_bert_spc:
                            postfix += 'ATE_F1: N.A. for LCF-ATEPC under use_bert_spc=True)'
                        else:
                            postfix += 'ATE_F1: {}(max:{})'.format(current_ate_test_f1, self.opt.max_test_metrics[
                                'max_ate_test_f1'])
                    else:
                        postfix = 'Epoch:{} | No evaluation until epoch:{}'.format(epoch, self.opt.evaluate_begin)

                    iterator.postfix = postfix
                    iterator.refresh()

        self.logger.info('-------------------------------------Training Summary-------------------------------------')
        self.logger.info(
            '  Max APC Acc: {:.5f} Max APC F1: {:.5f} Max ATE F1: {:.5f} Accumulated Loss: {}'.format(
                self.opt.max_test_metrics['max_apc_test_acc'],
                self.opt.max_test_metrics['max_apc_test_f1'],
                self.opt.max_test_metrics['max_ate_test_f1'],
                sum_loss)
        )
        self.logger.info('-------------------------------------Training Summary-------------------------------------')
        print('Training finished, we hope you can share your checkpoint with everybody, please see:',
              'https://github.com/yangheng95/PyABSA#how-to-share-checkpoints-eg-checkpoints-trained-on-your-custom-dataset-with-community')

        print_args(self.opt, self.logger)

        # return the model paths of multiple training
        # in case of loading the best model after training
        if save_path:
            return save_path
        else:
            # direct return model if do not evaluate
            if self.opt.model_path_to_save:
                save_path = '{0}/{1}_{2}/'.format(self.opt.model_path_to_save,
                                                  self.opt.model_name,
                                                  self.opt.lcf,
                                                  )
                save_model(self.opt, self.model, self.tokenizer, save_path)
            return self.model, self.opt, self.tokenizer, sum_apc_test_acc, sum_apc_test_f1, sum_ate_test_f1

    def evaluate(self, eval_ATE=True, eval_APC=True):
        apc_result = {'apc_test_acc': 0, 'apc_test_f1': 0}
        ate_result = 0
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        x2ms_adapter.x2ms_eval(self.model)
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}

        for i, batch in enumerate(self.eval_dataloader):
            input_ids_spc, segment_ids, input_mask, label_ids, polarity, \
            valid_ids, l_mask, lcf_cdm_vec, lcf_cdw_vec = batch

            input_ids_spc = x2ms_adapter.to(input_ids_spc, self.opt.device)
            input_mask = x2ms_adapter.to(input_mask, self.opt.device)
            segment_ids = x2ms_adapter.to(segment_ids, self.opt.device)
            valid_ids = x2ms_adapter.to(valid_ids, self.opt.device)
            label_ids = x2ms_adapter.to(label_ids, self.opt.device)
            polarity = x2ms_adapter.to(polarity, self.opt.device)
            l_mask = x2ms_adapter.to(l_mask, self.opt.device)
            lcf_cdm_vec = x2ms_adapter.to(lcf_cdm_vec, self.opt.device)
            lcf_cdw_vec = x2ms_adapter.to(lcf_cdw_vec, self.opt.device)
            ate_logits, apc_logits = self.model(input_ids_spc,
                                                token_type_ids=segment_ids,
                                                attention_mask=input_mask,
                                                labels=None,
                                                polarity=polarity,
                                                valid_ids=valid_ids,
                                                attention_mask_label=l_mask,
                                                lcf_cdm_vec=lcf_cdm_vec,
                                                lcf_cdw_vec=lcf_cdw_vec
                                                )
            if eval_APC:
                n_test_correct += x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.x2ms_sum((x2ms_adapter.argmax(apc_logits, -1) == polarity)))
                n_test_total += len(polarity)

                if test_polarities_all is None:
                    test_polarities_all = polarity
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = x2ms_adapter.cat((test_polarities_all, polarity), dim=0)
                    test_apc_logits_all = x2ms_adapter.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_ATE:
                if not self.opt.use_bert_spc:
                    label_ids = self.model.get_batch_token_labels_bert_base_indices(label_ids)
                ate_logits = x2ms_adapter.argmax(x2ms_adapter.nn_functional.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(ate_logits))
                label_ids = x2ms_adapter.tensor_api.numpy(x2ms_adapter.to(label_ids, 'cpu'))
                input_mask = x2ms_adapter.tensor_api.numpy(x2ms_adapter.to(input_mask, 'cpu'))
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(self.label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_APC:
            test_acc = n_test_correct / n_test_total

            test_f1 = x2ms_np.sklearn_metrics_f1_score(x2ms_adapter.argmax(test_apc_logits_all, -1), test_polarities_all,
                               labels=list(range(self.opt.polarities_dim)), average='macro')

            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'apc_test_acc': test_acc, 'apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = x2ms_adapter.tensor_api.split(report)
            ate_result = round(float(tmps[7]) * 100, 2)
        return apc_result, ate_result


# @retry
def train4atepc(opt, from_checkpoint_path, logger):
    # in case of handling ConnectionError exception
    trainer = Instructor(opt, logger)
    load_checkpoint(trainer, from_checkpoint_path)

    return trainer.run()
