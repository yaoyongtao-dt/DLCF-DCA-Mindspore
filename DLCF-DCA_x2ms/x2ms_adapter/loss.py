#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import mindspore
import mindspore.nn


def legacy_parameter(size_average, reduce, reduction):
    if size_average is None and reduce is None:
        return reduction

    size_average = True if size_average is None else size_average
    reduce = True if reduce is reduce is None else reduce

    if reduce:
        return 'mean' if size_average else 'sum'
    else:
        return 'none'


class CrossEntropyLoss(mindspore.nn.LossBase):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', sparse=True):
        if weight is not None:
            raise NotImplementedError(f"Unsupported CrossEntropyLoss with weight parameter")

        self.reduction = legacy_parameter(size_average, reduce, reduction)

        super(CrossEntropyLoss, self).__init__(reduction=self.reduction)
        self.ignore_index = ignore_index
        self.sparse = sparse
        self.on_value = mindspore.Tensor(1.0, dtype=mindspore.float32)
        self.off_value = mindspore.Tensor(0., dtype=mindspore.float32)
        self.one_hot = mindspore.ops.OneHot()
        self.softmax_cross_entropy = mindspore.ops.SoftmaxCrossEntropyWithLogits()
        self.sparse_softmax_cross_entropy = mindspore.ops.SparseSoftmaxCrossEntropyWithLogits()
        self.to_float(mindspore.float32)

    @staticmethod
    def _reshape_input(labels, logits):
        logits_shape, labels_shape = logits.shape, labels.shape
        reshape = ()
        if len(logits_shape) == 4:
            # shape of pytorch logits: (N, C, H, W) -> mindspore (N, C)
            # shape of pytorch labels: (N, H, W) -> mindspore (N,)
            logits = mindspore.ops.Transpose()(logits, (0, 2, 3, 1))
            new_n = logits_shape[0] * logits_shape[2] * logits_shape[3]
            reshape = (logits_shape[0], logits_shape[2], logits_shape[3])
            logits_shape = (new_n, logits_shape[1])
            labels_shape = (new_n,)
        logits = logits.reshape(logits_shape)
        labels = labels.reshape(labels_shape)
        return labels, logits, reshape

    def construct(self, input, target):
        if target.dtype not in (mindspore.int32, mindspore.int64):
            target = target.astype(mindspore.int32)

        self._dim_check(target, input)

        target, input, reshape = self._reshape_input(target, input)

        if self.sparse:
            if self.reduction == 'mean' and self.ignore_index == -100:
                return self.sparse_softmax_cross_entropy(input, target)
            onehot_labels = self.one_hot(target, input.shape[1], self.on_value, self.off_value)
        else:
            onehot_labels = target

        return self._masked_calculate(target, input, onehot_labels, reshape)

    def _dim_check(self, labels, logits):
        logits_ndim, labels_ndim = logits.ndim, labels.ndim
        if self.ignore_index != -100 and not (logits_ndim == 2 and labels_ndim <= 2):
            raise NotImplementedError(f"Unsupported CrossEntropyLoss input dim with ignore_index parameter: "
                                      f"logits {logits_ndim}, labels {labels_ndim}")
        elif (logits_ndim == 2 and labels_ndim <= 2) or (logits_ndim == 4 and labels_ndim == 3):
            pass
        else:
            raise NotImplementedError(f"Unsupported CrossEntropyLoss input dim: "
                                      f"logits {logits_ndim}, labels {labels_ndim}")
        return

    def _masked_calculate(self, labels, logits, onehot_labels, reshape):
        onehot_labels = onehot_labels.astype(logits.dtype)
        adjust_ratio = 1
        if self.ignore_index != -100:
            mask = (labels != self.ignore_index).astype(mindspore.float32)
            if self.reduction == 'mean':
                counted_bach = mask.sum().asnumpy().item()
                if counted_bach == 0:
                    return mindspore.Tensor([0])
                else:
                    adjust_ratio = mask.shape[0] / counted_bach
            mask = mask.reshape((-1, 1))
            onehot_labels = mindspore.ops.mul(mask, onehot_labels)
        loss = self.softmax_cross_entropy(logits, onehot_labels)[0]
        if reshape and self.reduction == 'none':
            loss = loss.reshape(reshape)
        return self.get_loss(loss, adjust_ratio)


class BCELoss(mindspore.nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        reduction = legacy_parameter(size_average, reduce, reduction)
        super().__init__(reduction=reduction, weight=weight)
        self.to_float(mindspore.float32)

    def construct(self, logits, labels):
        if logits.shape != labels.shape:
            logits_sqz = logits.squeeze()
            labels_sqz = labels.squeeze()
            if logits_sqz.shape != labels_sqz.shape:
                raise ValueError(f"In BCELoss, dimensions of 'logits' and 'labels' must be equal, but got "
                        f"dimension of 'logits' {logits.dim()} and dimension of 'labels' {labels.dim()}.")
            else:
                return super().construct(logits_sqz, labels_sqz)
        else:
            return super(BCELoss, self).construct(logits, labels)


class BCEWithLogitsLoss(mindspore.nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(reduction=reduction, weight=weight, pos_weight=pos_weight)
        self.to_float(mindspore.float32)


class KLDivLoss(mindspore.ops.KLDivLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', log_target=False):
        super().__init__(reduction=reduction)


class L1Loss(mindspore.nn.L1Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        reduction = legacy_parameter(size_average, reduce, reduction)
        super().__init__(reduction=reduction)


class SmoothL1Loss(mindspore.nn.SmoothL1Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0):
        self.reduction = legacy_parameter(size_average, reduce, reduction)
        if self.reduction not in ('sum', 'mean', 'none'):
            raise ValueError(f"Parameter reduction only support 'sum' or 'mean' or 'none', but got {self.reduction}")
        self.version_flag = mindspore.__version__ >= '1.8'
        if self.version_flag:
            super(SmoothL1Loss, self).__init__(beta=beta, reduction=self.reduction)
        else:
            super(SmoothL1Loss, self).__init__(beta=beta)

    def construct(self, logits, labels):
        if self.version_flag:
            return super(SmoothL1Loss, self).construct(logits, labels)
        loss = super(SmoothL1Loss, self).construct(logits, labels)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
