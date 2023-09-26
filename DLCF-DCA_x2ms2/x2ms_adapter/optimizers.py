#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from typing import Iterator
from types import GeneratorType
from collections import namedtuple

import mindspore.nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from .context import x2ms_context

OptimizerInfo = namedtuple('OptimizerInfo', ['instance', 'func_caller'])


class OptimAdaptorMixIn:
    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class Adam(mindspore.nn.Adam, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.Adam.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class SGD(mindspore.nn.SGD, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.SGD.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class RMSprop(mindspore.nn.RMSProp, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.RMSProp.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class Adagrad(mindspore.nn.Adagrad, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.Adagrad.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class AdamW(mindspore.nn.AdamWeightDecay, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.AdamWeightDecay.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class ASGD(mindspore.nn.ASGD, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.ASGD.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class FuncCaller:
    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def get_call(self, *args, **kwargs):
        args = (*args, self._args)
        self._kwargs.update(kwargs)
        return self._func(*args, **self._kwargs)


def _parse_params(params):
    parse_keys = ['params', 'lr', 'weight_decay', 'order_params', 'grad_centralization']
    new_params = []
    for param in params:
        new_param = {}
        for key in param.keys():
            if isinstance(param[key], Iterator):
                param[key] = list(param[key])
            if key in parse_keys:
                new_param[key] = param[key]
        new_params.append(new_param)
    return new_params


def params_dict_to_list(params):
    if isinstance(params[0], dict):
        new_params = _parse_params(params)
        return new_params
    return params


class OptimRegister:
    def __init__(self):
        self._func = None
        self._register_info = []
        self._lr_scheduler = None

    @staticmethod
    def _params_to_list(params):
        if isinstance(params, (GeneratorType, Iterator)):
            params = list(params)
        return params

    def adam(self, params, lr=0.001, betas=(0.9, 0.999),
             eps=1e-8, weight_decay=0, amsgrad=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        optimizer_instance = Adam(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(Adam, *params, **kwargs)))
        return optimizer_instance

    def sgd(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "dampening": dampening,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
        }
        optimizer_instance = SGD(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(SGD, *params, **kwargs)))
        return optimizer_instance

    def rmsprop(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.0, centered=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "epsilon": eps,
            "centered": centered,
            "weight_decay": weight_decay,
        }
        optimizer_instance = RMSprop(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(RMSprop, *params, **kwargs)))
        return optimizer_instance

    def adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "accum": float(initial_accumulator_value) + eps
        }
        optimizer_instance = Adagrad(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(Adagrad, *params, **kwargs)))
        return optimizer_instance

    def adamw(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        }
        optimizer_instance = AdamW(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(AdamW, *params, **kwargs)))
        return optimizer_instance

    def asgd(self, params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "lambd": lambd,
            "alpha": alpha,
            "t0": t0,
            "weight_decay": weight_decay
        }
        optimizer_instance = ASGD(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance, FuncCaller(AdamW, *params, **kwargs)))
        return optimizer_instance

    def adadelta(self, params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        """
        MindSpore only provide mindspore.ops.ApplyAdadelta, which is not a fully functional optimizer class
        """
        raise NotImplementedError('Currently Adadelta optimizer is not supported.')

    def adamax(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        """
        MindSpore only provide mindspore.ops.ApplyAdaMax, which is not a fully functional optimizer class
        """
        raise NotImplementedError('Currently Adamax optimizer is not supported.')

    def get_instance(self):
        if len(self._register_info) == 0:
            raise RuntimeError('No optimizer instance has been created.')
        elif len(self._register_info) > 1:
            return ConcatOptimizer(list(optimizer_info.instance for optimizer_info in self._register_info))
        return self._register_info[-1].instance


def _record_args(optimizer, kwargs, params):
    if hasattr(optimizer, 'x2ms_input_kwargs'):
        return
    optimizer.x2ms_input_kwargs = kwargs
    if isinstance(params[0], dict):
        optimizer.x2ms_param_list = _list(params)
    else:
        optimizer.x2ms_param_list = [{'params': params}]
    if 'learning_rate' in kwargs:
        optimizer.initial_lr = kwargs['learning_rate']


class ConcatOptimizer(mindspore.nn.Optimizer):
    def __init__(self, optimizer_list):
        parameters = ()
        for optimizer in optimizer_list:
            parameters += optimizer.parameters
        super().__init__(learning_rate=0.1, parameters=parameters, weight_decay=0.0, loss_scale=1.0)
        self.optimizer_list = optimizer_list

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        success = ()
        start = 0
        for optimizer in self.optimizer_list:
            success += optimizer(gradients[start:(start + len(optimizer.parameters))])
            start = start + len(optimizer.parameters)
        return success


def create_param_groups_modifiers(optim):
    param_list = []
    for index, params in enumerate(optim.x2ms_param_list):
        param_list.append(OptimizerParamGroupsModifier(optim, params, index))
    return param_list


class OptimizerParamGroupsModifier:
    def __init__(self, optimizer, param, index=0):
        self.index = index
        self._optimizer = optimizer
        self.param_dict = dict(param)
        if 'lr' not in self.param_dict:
            self.param_dict['lr'] = optimizer.initial_lr
        if hasattr(optimizer, 'momentum'):
            if isinstance(optimizer.momentum, mindspore.Tensor):
                self.param_dict['momentum'] = float(optimizer.momentum.asnumpy())
            else:
                self.param_dict['momentum'] = optimizer.momentum

    def __setitem__(self, key, value):
        if key == 'lr':
            self.set_lr(value)
        elif key == 'momentum':
            self.set_momentum(value)
        else:
            self.param_dict[key] = value

    def __getitem__(self, key):
        if key == 'momentum' and hasattr(self._optimizer, 'momentum'):
            _momentum = self._optimizer.momentum
            return float(_momentum.asnumpy()) if isinstance(_momentum, mindspore.Tensor) else _momentum
        else:
            return self.param_dict.get(key)

    def __iter__(self):
        return iter(self.param_dict)

    def setdefault(self, key, default=None):
        self.param_dict.setdefault(key, default)

    def set_lr(self, value):
        if self._optimizer.is_group_lr:
            self._optimizer.learning_rate[self.index].set_data(Tensor(value, mstype.float32))
        else:
            self._optimizer.learning_rate.set_data(Tensor(value, mstype.float32))
        self.param_dict['lr'] = value

    def set_momentum(self, value):
        if hasattr(self._optimizer, 'momentum'):
            if isinstance(self._optimizer.momentum, mindspore.Tensor):
                self._optimizer.momentum.assign_value(mindspore.Tensor(value, mindspore.float32))
            else:
                self._optimizer.momentum = value
            self.param_dict['momentum'] = value


class _RequiredMindsporeCellParameter(object):
    def __repr__(self):
        return "<required parameter>"


@property
def get_param_groups(self):
    if hasattr(self, 'x2ms_param_groups'):
        return self.x2ms_param_groups
    return []


def _list(param):
    return param if isinstance(param, list) else [param]


def add_param_group(self, param_group):
    if 'lr' not in param_group:
        param_group['lr'] = self.initial_lr
    self.x2ms_param_list += _list(param_group)
    self.__init__(self.x2ms_param_list, **self.x2ms_input_kwargs)


mindspore.nn.Optimizer.param_groups = get_param_groups
mindspore.nn.Optimizer.add_param_group = add_param_group
optim_register = OptimRegister()
required = _RequiredMindsporeCellParameter()
