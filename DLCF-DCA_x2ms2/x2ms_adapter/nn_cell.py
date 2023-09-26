#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from collections import OrderedDict
import mindspore
import mindspore.nn

from .decorator import x2ms_func_decorator


@x2ms_func_decorator(mindspore.nn.Cell)
def apply(obj, fn):
    fn(obj)
    for _, cell in obj.cells_and_names():
        fn(cell)


@x2ms_func_decorator(mindspore.nn.Cell)
def children(obj):
    return obj.cells()


@x2ms_func_decorator(mindspore.nn.Cell)
def modules(obj):
    return (m[1] for m in obj.cells_and_names())


@x2ms_func_decorator(mindspore.nn.Cell)
def named_children(obj):
    return obj.name_cells().items()


@x2ms_func_decorator(mindspore.nn.Cell)
def register_buffer(obj, name, tensor):
    if tensor is not None:
        setattr(obj, name, mindspore.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, name, None)


@x2ms_func_decorator(mindspore.nn.Cell)
def register_forward_hook(obj, hook):
    original_construct = obj.construct

    class ForwardHookHandler:
        def __init__(self, module, original_construct):
            self.module = module
            self.original_construct = original_construct

        def remove(self):
            self.module.construct = self.original_construct

    def new_construct(self, *args):
        inputs = args
        outputs = original_construct(self, *inputs)
        hook(self, inputs, outputs)

        return outputs

    obj.construct = new_construct

    return ForwardHookHandler(obj, original_construct)


@x2ms_func_decorator(mindspore.nn.Cell)
def zero_grad(*args, **kwargs):
    pass


@x2ms_func_decorator(mindspore.nn.Cell)
def private_load_from_state_dict(obj, state_dict, prefix, local_metadata, strict,
                                 missing_keys, unexpected_keys, error_msgs):
    """
    Currently not implemented.
    """
    pass


@property
def _modules(self):
    return OrderedDict(self.name_cells())
