#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import itertools
import statistics
import numbers
import math
from functools import reduce
from typing import NamedTuple
import mindspore
import mindspore.ops as ops
import mindspore.numpy as ms_np
import numpy as np

from scipy.special import erfinv
from .decorator import x2ms_func_decorator
from . import torch_base_api
from .numpy import TensorNumpy

classic_tensor_format = mindspore.Tensor.__format__
classic_tensor_getitem = mindspore.Tensor.__getitem__
classic_tensor_setitem = mindspore.Tensor.__setitem__

NamedtupleValuesIndices = NamedTuple("namedtuple_values_indices",
                                     [("values", mindspore.Tensor), ("indices", mindspore.Tensor)])


def tensor_format(self, format_spec):
    if self.dim() > 0:
        return classic_tensor_format(self, format_spec)
    return self.asnumpy().item().__format__(format_spec)


def slice_convert(index):
    new_index_start, new_index_stop, new_index_step = index.start, index.stop, index.step
    if not isinstance(index.start, (int, mindspore.Tensor)) and index.start is not None:
        new_index_start = index.start.item()
    if not isinstance(index.stop, (int, mindspore.Tensor)) and index.stop is not None:
        new_index_stop = index.stop.item()
    if not isinstance(index.step, (int, mindspore.Tensor)) and index.step is not None:
        new_index_step = index.step.item()
    index = slice(new_index_start, new_index_stop, new_index_step)
    return index


def convert_index(index):
    if isinstance(index, slice):
        index = slice_convert(index)
    if isinstance(index, range):
        index = list(index)
    if isinstance(index, tuple):
        new_index = []
        for each_index in index:
            if isinstance(each_index, slice):
                index_element = slice_convert(each_index)
                new_index.append(index_element)
            elif isinstance(each_index, range):
                new_index.append(list(each_index))
            else:
                new_index.append(each_index)
        index = tuple(new_index)
    if isinstance(index, np.ndarray):
        index = index.item()
    return index


def tensor_getitem(self, index):
    new_index = convert_index(index)
    if _check_empty_all_index(index):
        return mindspore.ops.Zeros()(_get_index_shape(self, index), self.dtype)
    if isinstance(new_index, mindspore.Tensor) and new_index.dtype == mindspore.bool_:
        return classic_tensor_getitem(
            self.reshape(-1, *self.shape[new_index.dim():]), new_index.reshape(-1).asnumpy().tolist())
    if isinstance(new_index, tuple) and any(is_bool_tensor(obj) for obj in new_index):
        new_shape, new_index = _get_shape_index(self, new_index)
        if is_bool_tensor(self):
            self = self.astype(mindspore.float32)
            return classic_tensor_getitem(self.reshape(*new_shape), new_index).astype(mindspore.bool_)
        return classic_tensor_getitem(self.reshape(*new_shape), new_index)
    return classic_tensor_getitem(self, new_index)


def _get_shape_index(tensor, index):
    new_index = list(index)
    shape = list(tensor.shape)
    new_shape = []
    for i, items in enumerate(index):
        dim = 1
        if is_bool_tensor(items):
            dim = new_index[i].dim()
            new_index[i] = new_index[i].reshape(-1).asnumpy().tolist()
        if dim > 1:
            new_shape.append(reduce(lambda x, y: x * y, shape[:dim]))
        else:
            new_shape.append(shape[0])
        shape = shape[dim:]
    if shape:
        new_shape.extend(shape)
    new_index = tuple(new_index)
    return new_shape, new_index


def is_bool_tensor(obj):
    return isinstance(obj, mindspore.Tensor) and obj.dtype == mindspore.bool_


def _get_index_shape(tensor, index):
    if isinstance(index, tuple):
        return _get_tuple_index_shape(tensor, index)
    if isinstance(index, list):
        return (len(index), *tensor.shape[1:])
    if isinstance(index, mindspore.Tensor):
        return (0, *tensor.shape[index.dim():])
    raise NotImplementedError("Not supported index type of tensor: ", type(index))


def _get_tuple_index_shape(tensor, index):
    tensor_index = 0
    shape = []
    for i in index:
        if isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else tensor.shape[tensor_index]
            step = i.step if i.step is not None else 1
            if start >= stop:
                shape.append(0)
            else:
                shape.append(math.ceil((stop - start) / step))
        elif isinstance(i, int):
            pass
        elif isinstance(i, (tuple, list)):
            shape.append(len(i))
        elif isinstance(i, mindspore.Tensor):
            shape.append(_get_tensor_index_shape(i))
            if i.dim() != 0:
                tensor_index += i.dim()
                continue
        elif i is None:
            shape.append(1)
            continue
        else:
            raise NotImplementedError("Not supported index type of tensor: ", type(i))
        tensor_index += 1
    shape.extend(tensor.shape[tensor_index:])
    return tuple(shape)


def _get_tensor_index_shape(index):
    if index.dtype == mindspore.bool_:
        return int(index.astype(mindspore.float32).sum().asnumpy().item())
    return index.size


def _check_empty_all_index(index):
    if isinstance(index, tuple):
        for sub_index in index:
            if _check_empty_index(sub_index):
                return True

    return _check_empty_index(index)


def _check_empty_index(index):
    if isinstance(index, (list, tuple)):
        return len(index) == 0
    if isinstance(index, mindspore.Tensor):
        return _check_empty_tensor_index(index)
    return False


def _check_empty_tensor_index(index):
    return index.size == 0 or (index.dtype == mindspore.bool_ and not index.any())


_TENSOR_DTYPE_RANGE = {str(mindspore.float16): (-65504, 65504), str(mindspore.int16): (-32768, 32767),
                       str(mindspore.float32): (-3.4*10**38, 3.4*10**38), str(mindspore.int32): (-2**31, 2**31),
                       str(mindspore.float64): (-1.7*10**308, 1.7*10**308), str(mindspore.int64): (-2**63, 2**63-1)
                       }


def _value_mapping(dtype_name):
    if dtype_name not in _TENSOR_DTYPE_RANGE.keys():
        raise ValueError('Not implement dtype mapping')
    return _TENSOR_DTYPE_RANGE.get(dtype_name)


def _is_setitem_supported_tensor_type(dtype):
    if mindspore.context.get_context('device_target') == 'Ascend':
        return dtype in (mindspore.float32, mindspore.float16, mindspore.bool_)
    else:
        return dtype not in (mindspore.uint8, mindspore.int8)


def _reshape_format(target, index):
    target_shape = list(target.shape)
    reshape_list = []
    new_index = []
    dim = 0
    reshape_flag = False
    for one_idx in index:
        if isinstance(one_idx, mindspore.Tensor) and one_idx.dtype == mindspore.bool_:
            reshape_list.append(-1)
            dim += one_idx.dim()
            new_index.append(one_idx.reshape(-1).asnumpy().tolist())
            reshape_flag = True
        else:
            reshape_list.append(target_shape[dim])
            new_index.append(one_idx)
            dim += 1
    if dim != target.dim():
        reshape_list.extend(target_shape[dim:])
    if reshape_flag:
        return tuple(new_index), reshape_list
    else:
        return tuple(new_index), []


def _update_value(target, index, value):
    if isinstance(index, tuple):
        new_index, reshape_list = _reshape_format(target, index)
        if reshape_list:
            target_r = target.reshape(reshape_list)
            classic_tensor_setitem(target_r, new_index, value)
            target.assign_value(target_r.reshape(target.shape))
        else:
            classic_tensor_setitem(target, new_index, value)
    else:
        classic_tensor_setitem(target, index, value)


def tensor_setitem(self, index, value):
    if isinstance(value, float):
        dtype = str(self.dtype)
        if value == math.inf:
            value = _value_mapping(dtype)[1]
        elif value == -math.inf:
            value = _value_mapping(dtype)[0]

    index = convert_index(index)
    old_type = self.dtype

    if isinstance(index, mindspore.Tensor):
        if index.size == 0:
            return
        elif index.dtype == mindspore.bool_:
            if index.dim() != 1:
                index = tuple([index])
            else:
                index = index.reshape(-1).asnumpy().tolist()

    if not isinstance(value, mindspore.Tensor):
        value = mindspore.Tensor(value)

    if _is_setitem_supported_tensor_type(self.dtype):
        _update_value(self, index, value)
    else:
        converted = self.astype(mindspore.float32)
        _update_value(converted, index, value.astype(mindspore.float32))
        self.assign_value(converted.astype(old_type))


@x2ms_func_decorator(mindspore.Tensor)
def t(input):
    dim = len(input.shape)
    perm = (1, 0)
    if dim == 1:
        perm = (0,)
    return ops.transpose(input, perm)


@x2ms_func_decorator(mindspore.Tensor)
def topk(input, k, dim=-1, largest=True, sorted=True):
    return ops.TopK(sorted)(input, k)


@x2ms_func_decorator(mindspore.Tensor)
def eq(input, other):
    equal = ops.Equal()
    return equal(input, other)


@x2ms_func_decorator(mindspore.Tensor, mindspore.nn.Cell, np.ndarray)
def x2ms_float(obj):
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    if isinstance(obj, mindspore.Tensor):
        if obj.size == 0:
            return mindspore.ops.Zeros()(obj.shape, mindspore.float32)
        return obj.astype(mindspore.float32)
    else:
        return obj


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def permute(obj, *axis):
    if isinstance(obj, np.ndarray):
        return obj.transpose(*axis)
    return obj.transpose(*axis)


@x2ms_func_decorator(mindspore.Tensor, TensorNumpy)
def numpy(obj):
    if isinstance(obj, TensorNumpy):
        return np.array(obj)
    return obj.asnumpy()


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def contiguous(obj):
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def scatter_(obj, dim, index, src):
    def get_value(src_value, tensor_index):
        if isinstance(src_value, mindspore.Tensor):
            return src_value[tensor_index]
        if isinstance(src_value, numbers.Number):
            return src_value
        return src_value

    shape_tuple = index.shape
    numpy_index = index.asnumpy()
    list_range = []
    for shape in shape_tuple:
        list_range.append(range(shape))
    for tensor_idx_tuple in itertools.product(*list_range):
        idx_list = list(tensor_idx_tuple)
        idx_list[dim] = numpy_index[tuple(tensor_idx_tuple)].item()
        obj[tuple(idx_list)] = get_value(src, tuple(tensor_idx_tuple))

    return obj


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def unsqueeze(obj, dim):
    if isinstance(obj, np.ndarray):
        return np.expand_dims(obj, dim)
    expand_dims = mindspore.ops.ExpandDims()
    return expand_dims(obj, dim)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_type(obj, dtype=None, non_blocking=False, **kwargs):
    if dtype is None:
        return str(obj.dtype)
    if dtype == torch_base_api.LongTensor:
        return obj.astype(dtype=mindspore.int64)
    # Special processing is required when the character string type returned by x2ms_type is converted again.
    if dtype == torch_base_api.FloatTensor or dtype == 'Float32':
        return obj.astype(dtype=mindspore.float32)
    if isinstance(dtype, mindspore.common.Type):
        return obj.astype(dtype=dtype)
    raise NotImplementedError(f'Unsupported tensor dtype {dtype}')


@x2ms_func_decorator(mindspore.Tensor)
def add(input, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    return mindspore.ops.Add()(input, other)


@x2ms_func_decorator(mindspore.Tensor)
def sub(input, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    return mindspore.ops.Sub()(input, other)


@x2ms_func_decorator(mindspore.Tensor)
def mul(input, other):
    return mindspore.ops.Mul()(input, other)


@x2ms_func_decorator(mindspore.Tensor)
def exp(input, out=None):
    return mindspore.ops.Exp()(input)


@x2ms_func_decorator(mindspore.Tensor)
def div(self, other):
    return mindspore.ops.Div()(self, other)


@x2ms_func_decorator(mindspore.Tensor)
def add_(obj, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    result = mindspore.ops.Add()(obj, other)
    mindspore.ops.Assign()(obj, result)
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def sub_(obj, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    result = mindspore.ops.Sub()(obj, other)
    mindspore.ops.Assign()(obj, result)
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def mul_(obj, value):
    result = mindspore.ops.Mul()(obj, value)
    mindspore.ops.Assign()(obj, result)
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def div_(obj, value):
    result = mindspore.ops.Div()(obj, value)
    mindspore.ops.Assign()(obj, result)
    return obj


@property
def property_data(self):
    return self


@property
def property_device(self):
    return torch_base_api.Device()


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_sum(obj, *args, **kwargs):
    return torch_base_api.x2ms_sum(obj, *args, **kwargs)


def tensor_size(tensor, dim=None):
    if dim is None:
        return tensor.shape
    return tensor.shape[dim]


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def x2ms_size(obj, *args, **kwargs):
    return tensor_size(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor)
def item(obj, *args, **kwargs):
    return obj.asnumpy().item()


@x2ms_func_decorator(mindspore.Tensor)
def nelement(obj):
    return obj.size


def tensor_repeat(obj, *sizes):
    if isinstance(sizes[0], (tuple, list)):
        sizes = sizes[0]
    if obj.dtype == mindspore.bool_:
        tensor = obj.astype(mindspore.int32)
        return ms_np.tile(tensor, sizes) > 0
    return ms_np.tile(obj, sizes)


@x2ms_func_decorator(mindspore.Tensor)
def repeat(obj, *args, **kwargs):
    return tensor_repeat(obj, *args, **kwargs)


def tensor_mean(tensor, dim=None, keepdim=False):
    if tensor.dtype == mindspore.bool_:
        tensor = tensor.astype(mindspore.float32)
    return tensor.mean(axis=dim, keep_dims=keepdim)


@x2ms_func_decorator(mindspore.Tensor)
def mean(obj, *args, **kwargs):
    return tensor_mean(obj, *args, **kwargs)


def tensor_std(tensor, dim=None, unbiased=True, keepdim=False):
    return tensor.std(axis=dim, ddof=1, keepdims=keepdim)


@x2ms_func_decorator(mindspore.Tensor)
def std(obj, *args, **kwargs):
    return tensor_std(obj, *args, **kwargs)


def tensor_transpose(tensor, dim0, dim1):
    dim = tensor.dim()
    _dim0 = dim0 if dim0 >= 0 else (dim0 + dim)
    _dim1 = dim1 if dim1 >= 0 else (dim1 + dim)
    dim_list = list(range(dim))
    dim_list[_dim0] = _dim1
    dim_list[_dim1] = _dim0
    return tensor.transpose(*dim_list)


@x2ms_func_decorator(mindspore.Tensor)
def transpose(obj, *args, **kwargs):
    return tensor_transpose(obj, *args, **kwargs)


@property
def transpose_(obj):
    if obj.size == 0:
        return mindspore.ops.Zeros()(tuple(reversed(list(obj.shape))), obj.dtype)
    if obj.dtype == mindspore.bool_:
        return obj.astype(mindspore.float32).transpose().astype(mindspore.bool_)
    return obj.transpose()


@x2ms_func_decorator(mindspore.Tensor)
def copy_(obj, src, non_blocking=False):
    if obj.parent_tensor_ is not None:
        obj.parent_tensor_[obj.index_of_parent_] = src
        return obj.parent_tensor_[obj.index_of_parent_]
    else:
        mindspore.ops.Assign()(obj, src)
        return obj


@x2ms_func_decorator(mindspore.Tensor)
def sqrt_(obj):
    obj.assign_value(mindspore.ops.Sqrt()(obj))
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_int(obj):
    return obj.astype(mindspore.int32)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_bool(obj):
    return obj.astype(mindspore.bool_)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_all(obj, dim=None, keepdim=False):
    return obj.all(axis=dim, keep_dims=keepdim)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_any(obj, dim=None, keepdim=False):
    return obj.any(axis=dim, keep_dims=keepdim)


@x2ms_func_decorator(mindspore.Tensor)
def cumsum(obj, dim, dtype=None):
    return obj.cumsum(axis=dim, dtype=dtype)


@x2ms_func_decorator(mindspore.Tensor)
def minimum(obj, other, *, out=None):
    return torch_base_api.minimum(obj, other, out=out)


@x2ms_func_decorator(mindspore.Tensor)
def diagonal(obj, *args, **kwargs):
    return torch_base_api.x2ms_diagonal(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor)
def expand_as(obj, other):
    return expand(obj, other.shape)


@x2ms_func_decorator(mindspore.Tensor)
def reshape_as(obj, other):
    return obj.reshape(other.shape)


@x2ms_func_decorator(mindspore.Tensor)
def stride(obj, dim=None):
    shape = list(obj.shape)
    result = []
    for index in range(len(obj.shape)):
        if not shape[index + 1:]:
            shape.append(1)
        result.append(reduce(lambda x, y: x * y, shape[index + 1:]))
    return tuple(result)[dim] if dim is not None else tuple(result)


@x2ms_func_decorator(mindspore.Tensor)
def take(obj, index):
    return obj.take(indices=index)


@x2ms_func_decorator(mindspore.Tensor)
def mm(obj, *args, **kwargs):
    return torch_base_api.mm(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor)
def new_zeros(obj, *size, dtype=None, device=None, requires_grad=False):
    """
       The device parameter is not supported.
    """
    if device is not None:
        raise NotImplementedError('The device parameter is not supported.')
    if isinstance(size[0], tuple) or isinstance(size[0], list):
        size = tuple(size[0])
    if dtype is None:
        dtype = mindspore.ops.DType()(obj)
    data = ops.Zeros()(size, dtype)
    if not requires_grad:
        data = mindspore.ops.stop_gradient(data)
    return data


@property
def grad(self):
    return mindspore.numpy.zeros_like(self)


@x2ms_func_decorator(mindspore.Parameter)
def requires_grad_(obj, requires_grad=True):
    obj.requires_grad = requires_grad


@x2ms_func_decorator(mindspore.Tensor)
def norm(obj, p='fro', dim=None, keepdim=False, dtype=None):
    if p not in ['fro', 2]:
        raise TypeError("MindSpore currently only supports L2 paradigm.")
    if dim is not None:
        norm_ops = mindspore.nn.Norm(axis=dim, keep_dims=keepdim)
    else:
        norm_ops = mindspore.nn.Norm(keep_dims=keepdim)
    return norm_ops(obj)


@x2ms_func_decorator(mindspore.Tensor)
def masked_fill(obj, mask, value):
    broadcast_to = mindspore.ops.BroadcastTo(obj.shape)
    mask = mask.astype(mindspore.int32)
    reverse_mask = (mask == 0).astype(mindspore.int32)
    mask = broadcast_to(mask)
    reverse_mask = broadcast_to(reverse_mask)
    return obj * reverse_mask + mask * value


@x2ms_func_decorator(mindspore.Tensor)
def masked_fill_(obj, mask, value):
    return obj.assign_value(obj.masked_fill(mask, value))


def tensor_argmax(tensor, dim=None, keepdim=False):
    return tensor.argmax(axis=dim)


@x2ms_func_decorator(mindspore.Tensor)
def argmax(obj, *args, **kwargs):
    return tensor_argmax(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor)
def argmin(obj, dim=None, keepdim=False):
    """
    The keepdim parameter is not supported.
    """
    if keepdim is True:
        raise NotImplementedError('The keepdim parameter is not supported.')
    return obj.argmin(axis=dim)


@x2ms_func_decorator(mindspore.Tensor)
def sigmoid(self):
    return torch_base_api.sigmoid(self)


def tensor_max(tensor, dim=None, keepdim=False):
    old_dtype = None
    if tensor.dtype in (mindspore.float64, mindspore.int64, mindspore.int32, mindspore.bool_):
        old_dtype = tensor.dtype
        tensor = tensor.astype(mindspore.float32)
    if dim is None:
        output = tensor.max(axis=dim, keepdims=keepdim)
        if old_dtype is not None:
            output = output.astype(old_dtype)
        return output
    else:
        max_ops = mindspore.ops.ArgMaxWithValue(axis=dim, keep_dims=keepdim)
        index, output = max_ops(tensor)
        if old_dtype is not None:
            output = output.astype(old_dtype)
        return output, index


@x2ms_func_decorator(mindspore.Tensor, TensorNumpy)
def x2ms_max(obj, *args, **kwargs):
    if isinstance(obj, TensorNumpy):
        if len(args) >= 1 and isinstance(args[0], np.ndarray):
            return np.maximum(obj, *args, **kwargs)
        else:
            return torch_base_api.numpy_max(obj, *args, **kwargs)
    return tensor_max(obj, *args, **kwargs)


def tensor_min(tensor, dim=None, keepdim=False):
    old_dtype = None
    if tensor.dtype in (mindspore.float64, mindspore.int64, mindspore.int32, mindspore.bool_):
        old_dtype = tensor.dtype
        tensor = tensor.astype(mindspore.float32)
    if dim is None:
        output = tensor.min(axis=dim, keepdims=keepdim)
        if old_dtype is not None:
            output = output.astype(old_dtype)
        return output
    else:
        min_ops = mindspore.ops.ArgMinWithValue(axis=dim, keep_dims=keepdim)
        index, output = min_ops(tensor)
        if old_dtype is not None:
            output = output.astype(old_dtype)
        return NamedtupleValuesIndices(output, index)


@x2ms_func_decorator(mindspore.Tensor, TensorNumpy)
def x2ms_min(obj, *args, **kwargs):
    if isinstance(obj, TensorNumpy):
        if len(args) >= 1 and isinstance(args[0], np.ndarray):
            return np.minimum(obj, *args, **kwargs)
        else:
            return torch_base_api.numpy_min(obj, *args, **kwargs)
    return tensor_min(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def long(obj, memory_format=None):
    if isinstance(obj, np.ndarray):
        return obj.astype(np.int64)
    if obj.size == 0:
        return mindspore.ops.Zeros()(obj.shape, mindspore.int64)
    return obj.astype(mindspore.int64)


@x2ms_func_decorator(mindspore.Tensor)
def numel(obj):
    return obj.size


@x2ms_func_decorator(mindspore.Tensor)
def expand(obj, *sizes):
    if isinstance(sizes[0], tuple):
        sizes = sizes[0]

    # pytorch supports tensor as dim, need convert
    new_size = []
    for sz in sizes:
        if isinstance(sz, mindspore.Tensor):
            new_size.append(sz.asnumpy().item())
        else:
            new_size.append(sz)
    sizes = tuple(new_size)

    if -1 in sizes:
        return mindspore.ops.BroadcastTo(sizes)(obj)
    return mindspore.numpy.broadcast_to(obj, sizes)


@x2ms_func_decorator(mindspore.Tensor)
def flatten(obj, *args, **kwargs):
    return torch_base_api.flatten(obj, *args, **kwargs)


@x2ms_func_decorator(mindspore.Tensor)
def zero_(obj, *args, **kwargs):
    obj.assign_value(mindspore.ops.zeros_like(obj))
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def t_(obj, *args, **kwargs):
    t_tensor = mindspore.ops.transpose(obj, (1, 0))
    obj.assign_value(t_tensor)
    return obj


@property
def is_cuda(self):
    return mindspore.context.get_context('device_target') in ['GPU', 'Ascend']


def tensor_and(variable_x, variable_y):
    if variable_x.size == 0:
        return variable_x

    if variable_x.dtype == mindspore.bool_ and variable_y.dtype == mindspore.bool_:
        return mindspore.numpy.logical_and(variable_x, variable_y)
    else:
        raise NotImplementedError(f"And operation on Input tensors with dtype: "
                                  f"{variable_x.dtype} and {variable_y.dtype} is currently unsupported.")


@x2ms_func_decorator(mindspore.Tensor, TensorNumpy)
def view(obj, *shape):
    if isinstance(obj, TensorNumpy):
        if isinstance(shape[0], list):
            return obj.reshape(tuple(shape[0]))
        return obj.reshape(*shape)
    if isinstance(shape[0], list):
        return obj.view(tuple(shape[0]))
    return obj.view(*shape)


def matmul(obj, tensor2):
    return mindspore.ops.matmul(obj, tensor2)


@x2ms_func_decorator(mindspore.Tensor)
def fill_(obj, value):
    dtype = mindspore.ops.DType()(obj)
    shape = mindspore.ops.Shape()(obj)
    new_tensor = mindspore.ops.Fill()(dtype, shape, value)
    mindspore.ops.Assign()(obj, new_tensor)
    return obj


@x2ms_func_decorator(mindspore.Tensor)
def flip(obj, *args, **kwargs):
    return tensor_flip(obj, *args, **kwargs)


def tensor_flip(tensor, dims):
    return mindspore.numpy.flip(tensor, dims)


@x2ms_func_decorator(mindspore.Tensor)
def erfinv_(obj, *args, **kwargs):
    return tensor_inplace_erfinv_(obj, *args, **kwargs)


def tensor_inplace_erfinv_(tensor):
    if mindspore.context.get_context('device_target') == 'Ascend':
        tensor.assign_value(mindspore.ops.Erfinv()(tensor))
        return tensor
    tensor.assign_value(mindspore.Tensor(erfinv(tensor.asnumpy()), dtype=mindspore.float32))
    return tensor


@x2ms_func_decorator(mindspore.Tensor)
def uniform_(obj, *args, **kwargs):
    return tensor_inplace_uniform_(obj, *args, **kwargs)


def tensor_inplace_uniform_(tensor, a=0.0, b=1.0):
    from_tensor = mindspore.Tensor(a, dtype=mindspore.float32)
    to_tensor = mindspore.Tensor(b, dtype=mindspore.float32)
    tensor.assign_value(mindspore.ops.uniform(tensor.shape, from_tensor, to_tensor))
    return tensor


@x2ms_func_decorator(mindspore.Tensor, TensorNumpy)
def clamp_(obj, *args, **kwargs):
    if isinstance(obj, TensorNumpy):
        return np.clip(obj, kwargs.get('min'), kwargs.get('max'), out=obj)
    return tensor_inplace_clamp_(obj, *args, **kwargs)


def tensor_inplace_clamp_(tensor, min=None, max=None):
    if tensor.size == 0:
        return tensor
    if tensor.parent_tensor_ is not None:
        new_tensor = mindspore.numpy.clip(tensor, min, max)
        tensor.parent_tensor_[tensor.index_of_parent_] = new_tensor
        return new_tensor
    tensor.assign_value(mindspore.numpy.clip(tensor, min, max))
    return tensor


@x2ms_func_decorator(mindspore.Tensor, np.ndarray)
def clamp(obj, min=None, max=None, out=None):
    if isinstance(obj, np.ndarray):
        return np.clip(obj, min, max, out=out)
    if out is None:
        return mindspore.numpy.clip(obj, min, max)
    else:
        return out.assign_value(mindspore.numpy.clip(obj, min, max))


def normal_(obj, *args, **kwargs):
    if isinstance(obj, mindspore.Tensor):
        return tensor_inplace_normal_(obj, *args, **kwargs)
    elif isinstance(obj, np.ndarray):
        return numpy_inplace_normal_(obj, *args, **kwargs)
    return obj.normal_(*args, **kwargs)


def numpy_inplace_normal_(np_array, mean=0, std=1, *, generator=None):
    np_array[:] = np.random.normal(mean, std, np_array.shape)
    return np_array


def tensor_inplace_normal_(tensor, mean=0, std=1, *, generator=None):
    tensor.assign_value(mindspore.ops.normal(tensor.shape,
                                             mindspore.Tensor(mean, dtype=mindspore.float32),
                                             mindspore.Tensor(std, dtype=mindspore.float32)))
    return tensor


@x2ms_func_decorator(mindspore.Tensor)
def floor_(obj, *args, **kwargs):
    return tensor_floor_(obj)


def tensor_floor_(tensor):
    tensor.assign_value(mindspore.ops.Floor()(tensor))
    return tensor


@x2ms_func_decorator(mindspore.Tensor)
def floor(obj):
    return mindspore.ops.Floor()(obj)


@x2ms_func_decorator(mindspore.Tensor)
def clone(obj, *args, **kwargs):
    return obj.copy()


@x2ms_func_decorator(mindspore.Tensor)
def tolist(obj):
    return obj.asnumpy().tolist()


@x2ms_func_decorator(mindspore.Tensor)
def softmax(self, dim):
    # dim in torch.Tensor.softmax() cannot be None
    return mindspore.ops.Softmax(dim)(self)


@x2ms_func_decorator(mindspore.Tensor)
def median(self):
    return mindspore.Tensor(statistics.median_low(self.asnumpy().flatten()))


@x2ms_func_decorator(mindspore.Tensor)
def argsort(obj, dim=-1, descending=False):
    return mindspore.ops.Sort(dim, descending)(obj)[1]


@x2ms_func_decorator(mindspore.Tensor)
def unique(obj, sorted=True, return_inverse=False, return_counts=False, dim=None):
    output = mindspore.numpy.unique(obj, return_inverse)
    if sorted:
        output = mindspore.ops.Sort()(output)[0]
    return output


@x2ms_func_decorator(np.ndarray)
def x2ms_dim(obj):
    """
    Only implemented for numpy
    """
    return obj.ndim


@x2ms_func_decorator(np.ndarray)
def index_fill_(obj, dim, index, val):
    if obj.ndim != index.ndim:
        NotImplementedError("index broadcast feature in torch is not implemented.")
    np.put_along_axis(obj, index, val, dim)


@x2ms_func_decorator(mindspore.Tensor)
def sort(obj, dim=-1, descending=False):
    """"
    input parameters stable=False, out=None in Pytorch V1.11 are not implemented
    """
    res = obj
    if obj.dtype != mindspore.float32 and obj.dtype != mindspore.float16:
        res = obj.astype(mindspore.float32)
    return mindspore.ops.Sort(axis=dim, descending=descending)(res)


@x2ms_func_decorator(mindspore.Tensor)
def tensor_or(variable_x, variable_y):
    if variable_x.dtype == mindspore.bool_ and variable_y.dtype == mindspore.bool_:
        return mindspore.numpy.logical_or(variable_x, variable_y)
    else:
        if mindspore.context.get_context('device_target') == 'Ascend':
            origin_type = variable_x.dtype
            converted_type = origin_type
            if origin_type in (mindspore.uint8, mindspore.int8):
                converted_type = mindspore.int32
            return mindspore.numpy.bitwise_or(variable_x.astype(converted_type),
                                              variable_y.astype(converted_type)).astype(origin_type)
        else:
            origin_type = variable_x.dtype
            result = mindspore.numpy.logical_or(variable_x.astype(mindspore.bool_), variable_y.astype(mindspore.bool_))
            return result.astype(origin_type)


@x2ms_func_decorator(mindspore.Tensor)
def sigmoid_(obj, *args, **kwargs):
    return tensor_sigmoid_(obj)


def tensor_sigmoid_(tensor):
    tensor.assign_value(mindspore.ops.Sigmoid()(tensor))
    return tensor


@x2ms_func_decorator(mindspore.Tensor)
def lt(input, other, *, out=None):
    return mindspore.ops.Less()(input, other)


@x2ms_func_decorator(mindspore.Tensor)
def unbind(tensor, dim=0):
    if tensor.size == 0:
        if dim < -1 * tensor.ndim or dim >= tensor.ndim:
            raise IndexError(f'Dim {dim} out of range [{-1 * tensor.ndim}, {tensor.ndim - 1}]')
        if dim < 0:
            dim += tensor.ndim
        sub_shape = tensor.shape[0:dim] + tensor.shape[dim + 1:]
        result = tuple(mindspore.ops.Zeros()(sub_shape, tensor.dtype) for _ in range(tensor.shape[dim]))
        return result
    return mindspore.ops.Unstack(dim)(tensor)


@x2ms_func_decorator(mindspore.Tensor)
def new_full(obj, size, fill_value, dtype=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = mindspore.ops.DType()(obj)
    data = mindspore.numpy.full(size, fill_value, dtype=dtype)
    if not requires_grad:
        data = mindspore.ops.stop_gradient(data)

    return data


@x2ms_func_decorator(mindspore.Tensor)
def split(tensor, split_size, dim=0):
    return torch_base_api.split(tensor, split_size, dim)


@x2ms_func_decorator(mindspore.Tensor)
def squeeze(tensor, dim=None):
    if dim is not None and tensor.shape[dim] != 1:
        return tensor

    if mindspore.ops.Size()(tensor) == 0:
        if dim is None:
            new_shape = [dim for dim in tensor.shape if dim != 1]
        else:
            new_shape = tensor.shape[:dim] + tensor.shape[dim + 1:]
        return mindspore.ops.Zeros()(tuple(new_shape), tensor.dtype)
    else:
        return tensor.squeeze(dim)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_round(input, decimals=0):
    if decimals != 0:
        raise NotImplementedError(f'MindSpore does not support round with decimals={decimals}.')
    return mindspore.ops.Round()(input)


@x2ms_func_decorator(mindspore.Tensor)
def prod(obj, dim=(), keepdim=False, *, dtype=None):
    if dtype:
        obj = obj.astype(dtype)
    return mindspore.ops.ReduceProd(keep_dims=keepdim)(obj, axis=dim)


@x2ms_func_decorator(mindspore.Tensor)
def sign(obj):
    return mindspore.numpy.sign(obj)


@x2ms_func_decorator(mindspore.Tensor, mindspore.nn.Cell)
def half(obj):
    if isinstance(obj, mindspore.nn.Cell):
        return obj.to_float(mindspore.float16)
    if obj.dtype in (mindspore.float64, mindspore.int64):
        return obj.astype(mindspore.float32)
    else:
        return obj.astype(mindspore.float16)


@x2ms_func_decorator(mindspore.Tensor)
def nonzero(obj, as_tuple=False):
    return torch_base_api.nonzero(obj, as_tuple=as_tuple)


@x2ms_func_decorator(mindspore.Tensor)
def tanh(obj):
    return mindspore.numpy.tanh(obj)


@x2ms_func_decorator(mindspore.Tensor)
def detach(obj):
    return mindspore.ops.stop_gradient(obj)


def parameter_iadd(self, other):
    self.set_data(self + other)
    return self


def parameter_imul(self, other):
    self.set_data(self * other)
    return self


def parameter_isub(self, other):
    self.set_data(self - other)
    return self


def parameter_idiv(self, other):
    if isinstance(other, numbers.Number) and other == 0:
        raise ValueError("Parameter divided by 0.")
    self.set_data(self / other)
    return self


@x2ms_func_decorator(mindspore.Tensor)
def type_as(obj, tensor):
    return obj.astype(tensor.dtype)


@x2ms_func_decorator(mindspore.Tensor)
def view_as(obj, tensor):
    return obj.view(tensor.shape)


@x2ms_func_decorator(mindspore.Tensor)
def chunk(input, chunks, dim=0):
    return torch_base_api.chunk(input, chunks, dim)


@x2ms_func_decorator(mindspore.Tensor)
def sqrt(obj, *, out=None):
    return mindspore.ops.Sqrt()(obj)


@x2ms_func_decorator(mindspore.Tensor)
def log(input, *, out=None):
    return mindspore.ops.Log()(input)


@x2ms_func_decorator(mindspore.Tensor)
def log2(obj):
    return torch_base_api.log2(obj)


@x2ms_func_decorator(mindspore.Tensor)
def double(obj):
    """
    npu do not support float64
    """
    return obj.astype(mindspore.float32)


@x2ms_func_decorator(mindspore.Tensor)
def x2ms_pow(obj, exponent, out=None):
    return torch_base_api.x2ms_pow(obj, exponent, out)


@x2ms_func_decorator(mindspore.Tensor)
def rsqrt(obj):
    return mindspore.ops.Rsqrt()(obj)
