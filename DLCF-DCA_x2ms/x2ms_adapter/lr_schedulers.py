#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import mindspore.nn
import numpy as np


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        super(_LRScheduler, self).__init__()
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.step()

    def step(self, step=None):
        self.last_epoch += 1
        for params, lr in zip(self.optimizer.param_groups, self.get_lr()):
            params['lr'] = mindspore.Tensor(lr, mindspore.float32)
        return list(param['lr'] for param in self.optimizer.param_groups)

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        return {}


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch != 0 and self.last_epoch % self.step_size == 0:
            return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)
        return list(param['lr'] for param in self.optimizer.param_groups)


class LambdaLR(_LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list((self.lr_lambda(self.last_epoch) * lr) for lr in self.base_lrs)


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        """
        Args:
            verbose currently unsupported
        """

        min_lr = float(eta_min)
        self.lr_group = [mindspore.nn.CosineDecayLR(min_lr, float(param_group['lr']), int(T_max))
                         for param_group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def construct(self, global_step):
        return self.get_lr()

    def get_lr(self):
        return [one_lr_schedule.construct(mindspore.Tensor(self.last_epoch)).asnumpy().item()
                for one_lr_schedule in self.lr_group]


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)
        return list(param['lr'] for param in self.optimizer.param_groups)


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        if isinstance(min_lr, (tuple, list)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.eps = eps
        self.verbose = verbose

        self.num_bad_epochs = None
        self.last_epoch = 0
        if mode == 'min':
            self.mode_worse = np.inf
        else:
            self.mode_worse = -np.inf
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        self.last_epoch += 1
        current_metrics = float(metrics)
        if self._is_better(current_metrics, self.best):
            self.num_bad_epochs = 0
            self.best = current_metrics
        else:
            self.num_bad_epochs += 1

        if self.cooldown > 0:
            self.num_bad_epochs = 0
            self.cooldown -= 1

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr <= self.eps:
                    continue
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {self.last_epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def _is_better(self, current, best):
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best * (1. - self.threshold)
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return current > best * (self.threshold + 1.)
            else:
                return current > best + self.threshold


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)
