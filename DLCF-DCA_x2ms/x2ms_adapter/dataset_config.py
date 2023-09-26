#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

DATASET_PADDING_CONFIG = {
        # Add the column name of dataset that requires padding and the pad value before dataloader
        # format: column_name: pad_value
        # For example: 'column_0': 0,
}
# Whether the data returned by the getitem function of the dataset is ndarray, default is False
DATASET_RETURN_NDARRAY = False

# Whether the data returned by the getitem function of the dataset is its original data type, default is False
# Usually this parameter should be set to True is collate_fn is applied with data type related operations
DATASET_RETURN_TYPE_FLAG = False
