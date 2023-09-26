#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import os
from typing import Optional, Callable, Any, List, Tuple

import numpy as np
import PIL.Image

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as v_transforms
import mindspore.dataset.transforms.py_transforms as transforms

from mindspore.communication.management import get_rank, get_group_size, context
from mindspore.dataset import MappableDataset, BatchDataset
from .context import x2ms_context
from .dataset_config import DATASET_PADDING_CONFIG, DATASET_RETURN_NDARRAY, DATASET_RETURN_TYPE_FLAG
from .util_api import np_to_tensor


def _dataset_len(self):
    self.dataset_size = None
    return self.get_dataset_size()


@property
def mindspore_dataset(self):
    return self.children[0]


@property
def dataset_classes(self):
    child_dataset = self
    while True:
        if isinstance(child_dataset, MappableDataset):
            break
        if not child_dataset.children:
            return []
        child_dataset = child_dataset.children[0]

    if isinstance(child_dataset, ds.Cifar10Dataset):
        return __read_meta(os.path.join(child_dataset.dataset_dir, "batches.meta.txt"))
    elif isinstance(child_dataset, ds.Cifar100Dataset):
        return __read_meta(os.path.join(child_dataset.dataset_dir, "fine_label_names.txt"))
    elif isinstance(child_dataset, ds.ImageFolderDataset):
        return os.listdir(child_dataset.dataset_dir)
    else:
        raise NotImplementedError("Cannot get classes from this dataset now.")


def __read_meta(meta_file_path):
    with open(meta_file_path, 'r') as meta_file:
        content = meta_file.read(1024 * 1024)
    return list(class_content for class_content in content.splitlines() if len(class_content.strip()) != 0)


@property
def get_transform(self):
    return self.operations


@get_transform.setter
def set_transform(self, transform_to_set):
    self.operations = [v_transforms.ToPIL()]
    if isinstance(transform_to_set, list):
        self.operations.extend(transform_to_set)
    if isinstance(transform_to_set, transforms.Compose):
        self.operations.extend(transform_to_set.transforms)
    self.operations.append(_ensure_numpy_array)
    self.operations = transforms.Compose.reduce(self.operations)


mindspore.dataset.Dataset.__len__ = _dataset_len
mindspore.dataset.Dataset.dataset = mindspore_dataset
mindspore.dataset.Dataset.classes = dataset_classes
mindspore.dataset.Dataset.transform = get_transform
mindspore.dataset.Dataset.transform = set_transform


class RawDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        x2ms_context.thread_start_transform()
        sample = dataset[0]
        if DATASET_RETURN_TYPE_FLAG:
            self.dataset_return_type_list = [type(i) for i in sample]
        x2ms_context.thread_end_transform()
        self.is_dict = isinstance(sample, dict)

        if self.is_dict:
            self.column_names = list(sample.keys())
            self.column_records = [1] * len(self.column_names)
        else:
            if not isinstance(sample, tuple):
                sample = (sample,)
            # column_records is used to record the number of tensor in each tuple
            self.column_names, self.column_records = self._generate_column_names(sample)

    def __getitem__(self, item):
        x2ms_context.thread_start_transform()
        item = item.item()
        output = self.dataset[item]
        output = self._flatten_dataset_item(output)
        output = tuple(self._to_numpy_array(value) for value in output)
        x2ms_context.thread_end_transform()
        return output

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_numpy_array(data):
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]
        if isinstance(data, np.ndarray):
            if data.dtype == np.int64:
                return data.astype(np.int32)
            if data.dtype == np.float64:
                return data.astype(np.float32)
            return data
        if isinstance(data, mindspore.Tensor):
            if data.dtype == mindspore.int64:
                return data.astype(mindspore.int32).asnumpy()
            if data.dtype == mindspore.float64:
                return data.astype(mindspore.float32).asnumpy()
            return data.asnumpy()
        else:
            result = np.asarray(data)
            if result.dtype == np.int64:
                return result.astype(np.int32)
            if result.dtype == np.float64:
                return result.astype(np.float32)
            if result.dtype == object:
                return np.array(0, np.float32)
        return result

    @staticmethod
    def _generate_column_names(sample: tuple):
        column_names = []
        column_records = []
        num = 0
        for item in sample:
            if isinstance(item, dict):
                column_names.extend(f'column_{num}_{key}' for key in item.keys())
                column_records.append(1)
                num += 1
            elif isinstance(item, tuple):
                for index in range(len(item)):
                    column_names.append(f'column_{num + index}')
                column_records.append(len(item))
                num += len(item)
            else:
                column_names.append(f'column_{num}')
                column_records.append(1)
                num += 1
        return column_names, column_records

    @staticmethod
    def _flatten_dataset_item(dataset_item):
        if not isinstance(dataset_item, tuple):
            dataset_item = (dataset_item,)

        flattened = []
        for data in dataset_item:
            if isinstance(data, tuple):
                flattened.extend(data)
            elif isinstance(data, dict):
                flattened.extend(data.values())
            else:
                flattened.append(data)
        return flattened


class BatchDatasetWrapper(mindspore.dataset.BatchDataset):
    def __init__(self, dataset: RawDatasetWrapper, batch_size=1):
        self._is_dict = dataset.is_dict
        self._column_names = dataset.column_names
        self.column_records = dataset.column_records
        if DATASET_RETURN_TYPE_FLAG:
            self.dataset_return_type_list = dataset.dataset_return_type_list
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        input_columns = None if not DATASET_PADDING_CONFIG else list(DATASET_PADDING_CONFIG.keys())
        per_batch_map = None if not DATASET_PADDING_CONFIG else self.per_batch_map
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names, shard_id=get_rank(),
                                                                num_shards=get_group_size(), shuffle=False),
                             batch_size=batch_size, input_columns=input_columns, per_batch_map=per_batch_map)
        else:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names, shuffle=False),
                             batch_size=batch_size, input_columns=input_columns, per_batch_map=per_batch_map)

    def __iter__(self):
        if self._is_dict:
            return self.create_dict_iterator(output_numpy=True)
        else:
            return self._create_iterator_wrapper(self.create_tuple_iterator(output_numpy=True), self._column_names)

    @staticmethod
    def per_batch_map(*args):
        batch_size = len(args[0])
        new_column_list = []
        for index, column_name in enumerate(DATASET_PADDING_CONFIG.keys()):
            pad_value = DATASET_PADDING_CONFIG.get(column_name)
            if pad_value is not None:
                try:
                    pad_value = float(pad_value)
                except ValueError as e:
                    raise ValueError(f'"{pad_value}" cannot convert to float') from e
            max_shape = [max([data.shape[i] for data in args[index]]) for i in range(args[index][0].ndim)]
            pad_list = []
            for i in range(batch_size):
                pad_width = [(0, max_shape[j] - args[index][i].shape[j]) for j in range(args[index][0].ndim)]
                pad_list.append(np.pad(args[index][i], pad_width, constant_values=pad_value))
            new_column_list.append(pad_list)
        return tuple(new_column_list)

    @staticmethod
    def _create_iterator_wrapper(raw_iter, dataset_column_names):
        class IteratorWrapper:
            def __init__(self, iterator, column_names):
                self.iterator = iterator
                self.column_names = column_names

            def __iter__(self):
                return self

            def __next__(self):
                next_data = next(self.iterator)
                reconstructed = self._reconstruct_dataset_item(next_data, self.column_names)
                if len(reconstructed) == 1:
                    reconstructed = reconstructed[0]
                return reconstructed

            @staticmethod
            def _reconstruct_dataset_item(flattened_item, column_names):
                reconstructed = {}
                origin_idx = 0
                for idx, item in enumerate(flattened_item):
                    column_name = column_names[idx]
                    if not column_name.startswith(f'column_{origin_idx}'):
                        origin_idx += 1
                    if column_name == f'column_{origin_idx}':
                        reconstructed.update({origin_idx: item})
                    else:
                        key = column_name[len(f'column_{origin_idx}_'):]
                        reconstructed.setdefault(origin_idx, {}).update({key: item})

                return tuple(reconstructed.values())

        return IteratorWrapper(raw_iter, dataset_column_names)


def _create_batch_dataset_wrapper(dataset, batch_size):
    wrapped_dataset = RawDatasetWrapper(dataset)
    wrapped_dataset = BatchDatasetWrapper(wrapped_dataset, batch_size=batch_size)
    return wrapped_dataset


def _is_cifar100(dataset):
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, MappableDataset):
            return isinstance(child_dataset, ds.Cifar100Dataset)
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return False


def _del_cifar100_column(col_1, col_2, col_3, batch_info):
    return col_1, col_2,


def _batch_dataset(dataset, batch_size):
    if _is_cifar100(dataset):
        return dataset.batch(batch_size, per_batch_map=_del_cifar100_column,
                             input_columns=['image', 'fine_label', 'coarse_label'],
                             output_columns=['image', 'label'])
    return dataset.batch(batch_size)


def _add_sampler(dataset, sampler):
    if sampler and not isinstance(sampler, DistributedSampler):
        old_sampler = dataset.sampler
        dataset.use_sampler(sampler)
        dataset.add_sampler(old_sampler)


def data_loader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=1, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None):
    """
    batch_sampler is partially implemented. Only batch_size in batch_sampler is mapped.
    """
    if batch_sampler is not None:
        sampler_batch_size = getattr(batch_sampler, 'batch_size', 1)
        batch_size = max(batch_size, sampler_batch_size)
    if not isinstance(dataset, mindspore.dataset.Dataset):
        dataset = _create_batch_dataset_wrapper(dataset, batch_size)
    else:
        dataset = _batch_dataset(dataset.__safe_deepcopy__({}), batch_size)
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, MappableDataset):
            child_dataset.shuffle_flag = shuffle
            _add_sampler(child_dataset, sampler)
            child_dataset.num_parallel_workers = num_workers
        if isinstance(child_dataset, BatchDataset):
            child_dataset.drop_remainder = drop_last
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return dataset


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=1,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None):
        self.x2ms_dataset = dataset
        num_workers = 1 if num_workers == 0 else num_workers
        self.batch_sampler = data_loader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.dataset_return_type_list = []
        if isinstance(self.batch_sampler, BatchDatasetWrapper):
            if DATASET_RETURN_TYPE_FLAG:
                self.dataset_return_type_list = self.batch_sampler.dataset_return_type_list
            self.column_records = self.batch_sampler.column_records
        else:
            self.column_records = None

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for batch in iter(self.batch_sampler):
            if isinstance(batch, (tuple, list)):
                batch = self._tuple_batch_to_tensor(batch)
            elif isinstance(batch, dict):
                batch = {k: (np_to_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
            else:
                batch = np_to_tensor(batch) if isinstance(batch, np.ndarray) else batch
            if self.column_records is not None and max(self.column_records) != 1:
                batch_data_list = []
                start_idx = 0
                for idx in self.column_records:
                    batch_data_list.append(batch[start_idx] if idx == 1 else batch[start_idx:start_idx + idx])
                    start_idx += idx
                batch = tuple(batch_data_list)
            if self.collate_fn is None:
                yield batch
            else:
                if not isinstance(batch, tuple):
                    batch = (batch,)
                data = self.collate_fn_iter(self._convert_string_tensor(batch))
                yield data

    @property
    def dataset(self):
        return self.x2ms_dataset

    @staticmethod
    def data_filter(data, column_name):
        if column_name not in DATASET_PADDING_CONFIG:
            return data
        if isinstance(data, mindspore.Tensor) and data.dim() == 2:
            new_label_list = []
            for line in data.asnumpy():
                if (line != DATASET_PADDING_CONFIG.get(column_name)).any():
                    new_label_list.append(line)
            if not new_label_list:
                return mindspore.ops.Zeros()((0, *line.shape), data.dtype)
            return mindspore.Tensor(np.array(new_label_list))
        elif isinstance(data, mindspore.Tensor) and data.dim() == 1:
            for i, column_data in enumerate(data):
                if column_data == DATASET_PADDING_CONFIG.get(column_name):
                    return data[0: i]
        return data

    @staticmethod
    def _tuple_batch_to_tensor(batch):
        new_batch = []
        for item in batch:
            if isinstance(item, np.ndarray):
                new_batch.append(np_to_tensor(item))
            elif isinstance(item, dict):
                new_batch.append({k: (np_to_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()})
            else:
                new_batch.append(item)
        return tuple(new_batch)

    @staticmethod
    def _convert_string_tensor(batch):
        def _is_string_tensor(data):
            return isinstance(data, mindspore.Tensor) and data.dtype == mindspore.string

        def _convert_dict_item(item):
            converted = {}
            for k, v in item.items():
                if _is_string_tensor(v):
                    converted.update({k: mindspore.dataset.text.to_str(v.asnumpy()).tolist()})
                else:
                    converted.update({k: v})
            return converted

        new_batch = []
        for item in batch:
            new_item = item
            if isinstance(item, dict):
                new_item = _convert_dict_item(item)
            else:
                if _is_string_tensor(item):
                    new_item = mindspore.dataset.text.to_str(item.asnumpy()).tolist()
            new_batch.append(new_item)

        return new_batch

    @staticmethod
    def _split_batch_item(batch_item, batch_idx, column_idx):
        if isinstance(batch_item, dict):
            data = {k: DataLoader.data_filter(v[batch_idx], f'column_{column_idx}_{k}')
                    for k, v in batch_item.items()}
            if DATASET_RETURN_NDARRAY:
                data = {k: v[batch_idx].asnumpy() for k, v in data.items()}
        else:
            data = DataLoader._get_one_batch_item(batch_item, batch_idx)
            data = DataLoader.data_filter(data, f'column_{column_idx}')
            if DATASET_RETURN_NDARRAY:
                data = data.asnumpy()
        return data

    @staticmethod
    def _tensor_type_transform(data, target_type):
        if target_type == int or target_type == float:
            return data.asnumpy().item()
        elif target_type == list:
            return data.asnumpy().tolist()
        elif target_type == np.ndarray:
            return data.asnumpy()
        else:
            return data

    @staticmethod
    def _get_one_batch_item(batch_item, batch_idx):
        if isinstance(batch_item, (tuple, list)) and batch_item and isinstance(batch_item[0], mindspore.Tensor):
            return tuple(tensor[batch_idx] for tensor in batch_item)
        return batch_item[batch_idx]

    def collate_fn_iter(self, batch):
        data = []
        if isinstance(batch[0], dict):
            real_batch_size = len(list(batch[0].values())[0])
        else:
            real_batch_size = len(batch[0])

        for batch_idx in range(real_batch_size):
            each_index_data = []
            for column_idx, batch_item in enumerate(batch):
                new_batch_item = self._split_batch_item(batch_item, batch_idx, column_idx)
                if self.dataset_return_type_list:
                    new_batch_item = self._tensor_type_transform(new_batch_item,
                                                                 self.dataset_return_type_list[column_idx])
                each_index_data.append(new_batch_item)
            if len(each_index_data) == 1:
                data.append(each_index_data[0])
            elif len(each_index_data) > 1:
                data.append(tuple(each_index_data))
            else:
                continue
        data = self.collate_fn(data)
        return data


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def subset_dataset(dataset, indices):
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, mindspore.dataset.MappableDataset):
            _add_sampler(child_dataset, mindspore.dataset.samplers.SubsetSampler(indices))
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return dataset


class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tuple(self._type_convert(tensor) for tensor in tensors)

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    @staticmethod
    def _type_convert(data):
        if data.dtype == mindspore.float64:
            return data.astype(mindspore.float32).asnumpy()
        else:
            return data.asnumpy()


def random_split(dataset, lengths, generator=None):
    if isinstance(dataset, mindspore.dataset.Dataset):
        return dataset.split(lengths, randomize=True)

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(np.arange(sum(lengths))).tolist()
    split_datasets = []
    offset = 0
    for length in lengths:
        split_datasets.append(Subset(dataset, indices[offset: offset + length]))
        offset += length

    return tuple(split_datasets)


def _ensure_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, PIL.Image.Image):
        return np.asarray(data)
    elif isinstance(data, mindspore.Tensor):
        return data.asnumpy()
    else:
        raise NotImplementedError(f'Unsupported data type {type(data)}')


def uint_to_int(data):
    if data.dtype == np.uint32:
        return data.astype(np.int32)
    return data


class ImageFolder:
    def __new__(cls, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            ms_dataset = ds.ImageFolderDataset(dataset_dir=root, shard_id=get_rank(), num_shards=get_group_size())
        else:
            ms_dataset = ds.ImageFolderDataset(dataset_dir=root)
        target_transform_to_add = [uint_to_int]
        ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
        transform_to_add = [v_transforms.Decode(), v_transforms.ToPIL()]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
        return ms_dataset


class CocoDetection:
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        raise NotImplementedError


def _folder_pil_loader(path):
    with open(path, 'rb') as img_file:
        img = PIL.Image.open(img_file)
        return img.convert('RGB')


def folder_default_loader(path):
    return _folder_pil_loader(path)


def cifar10(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test',
                                       shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
    transform_to_add = [v_transforms.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    return ms_dataset


def cifar100(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test',
                                        shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'fine_label')
    transform_to_add = [v_transforms.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    return ms_dataset


def mnist(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.MnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                     shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.MnistDataset(dataset_dir=root, usage='train' if train else 'test')

    if transform:
        transform_to_add = [lambda data: PIL.Image.fromarray(data.squeeze(-1), mode='L')]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')

    return ms_dataset


def start_transform(*data):
    x2ms_context.thread_start_transform()
    return data[0] if len(data) == 1 else data


def end_transform(*data):
    x2ms_context.thread_end_transform()
    return data[0] if len(data) == 1 else data


def _map_transform(ms_dataset, transform, transform_to_add, input_columns):
    if transform:
        if isinstance(transform, list):
            transform_to_add.extend(transform)
        if isinstance(transform, transforms.Compose):
            transform_to_add.extend(transform.transforms)
    transform_to_add.append(_ensure_numpy_array)
    transform_to_add = [start_transform, *transform_to_add, end_transform]
    ms_dataset = ms_dataset.map(operations=transform_to_add, input_columns=input_columns)
    return ms_dataset


class Sampler(ds.Sampler):
    def __iter__(self):
        pass


class DistributedSampler(ds.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(num_shards=get_group_size(), shard_id=get_rank(), shuffle=shuffle)
        else:
            super().__init__(num_shards=1, shard_id=0, shuffle=shuffle)


class RandomSampler(mindspore.dataset.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super().__init__(replacement=replacement, num_samples=num_samples)


class SequentialSampler(mindspore.dataset.SequentialSampler):
    def __init__(self, data_source):
        super().__init__()


class SubsetRandomSampler(mindspore.dataset.SubsetRandomSampler):
    def __init__(self, indices, generator=None):
        super().__init__(indices)


class VisionDataset:
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms_function: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms_function is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transform_function or transform/target_transform can "
                             "be passed as argument")

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms_function = StandardTransform(transform, target_transform)
        self.transforms = transforms_function

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __repr__(self):
        title = f"Dataset {self.__class__.__name__}"
        body = ["Number of datapoints: {}".format(self.__len__())]

        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.extend(self.extra_repr().splitlines())

        if getattr(self, "transforms"):
            body.append(repr(self.transforms))
        lines = [title] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def extra_repr():
        return ""

    @staticmethod
    def _format_transform_repr(transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        title = f"{head}{lines[0]}"
        body = ["{}{}".format(" " * len(head), line) for line in lines[1:]]
        return [title] + body


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body.extend(self._format_transform_repr(self.transform, "Transform: "))
        if self.target_transform is not None:
            body.extend(self._format_transform_repr(self.target_transform, "Target transform: "))

        return '\n'.join(body)

    @staticmethod
    def _format_transform_repr(transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        title = f"{head}{lines[0]}"
        body = ["{}{}".format(" " * len(head), line) for line in lines[1:]]
        return [title] + body
