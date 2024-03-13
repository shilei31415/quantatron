import logging
import torch.utils.data as torchdata
import operator
import torch
from typing import Any, Callable, Dict, List, Optional, Union

from quantatron.utils.registry import Registry
from quantatron.utils.logger import _log_api_usage
from quantatron.config import configurable
from quantatron.utils.comm import get_world_size
from quantatron.utils.env import seed_all_rng
from quantatron.utils.file_io import PathManager
from quantatron.utils.logger import _log_api_usage, log_first_n
from .common import DatasetFromList, MapDataset, ToIterableDataset

from .samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)

DATASET_REGISTRY = Registry("DATASET")  # noqa F401 isort:skip
DATASET_REGISTRY.__doc__ = """
Registry for dataset, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_dataset(cfg):
    """
    根据cfg建立dataset
    mode = ['train', 'test']
    """
    train_dataset = cfg.TRAIN_DATASET.NAME
    _log_api_usage("data." + train_dataset)
    train_dataset = DATASET_REGISTRY.get(train_dataset)(cfg.TRAIN_DATASET)

    test_dataset = cfg.TEST_DATASET.NAME
    _log_api_usage("data." + test_dataset)
    test_dataset = DATASET_REGISTRY.get(test_dataset)(cfg.TEST_DATASET)

    return train_dataset, test_dataset


def build_dataset_train(cfg):
    train_dataset = cfg.TRAIN_DATASET.NAME
    _log_api_usage("data." + train_dataset)
    train_dataset = DATASET_REGISTRY.get(train_dataset)(cfg.TRAIN_DATASET)
    return train_dataset


def build_dataset_test(cfg):
    test_dataset = cfg.TRAIN_DATASET.NAME
    _log_api_usage("data." + test_dataset)
    train_dataset = DATASET_REGISTRY.get(test_dataset)(cfg.TRAIN_DATASET)
    return train_dataset


def collate_fn(batch):
    keys = batch[0].keys()
    collated_data = {}
    for k in keys:
        collated_data[k] = torch.stack([sample[k] for sample in batch])
    return collated_data


def build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        *,
        num_workers=0,
        collate_fn=collate_fn,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
    )


def _train_loader_from_config(cfg, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = build_dataset_train(cfg)
        _log_api_usage("dataset." + cfg.TRAIN_DATASET.NAME)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
                )
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.DATA_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(
        dataset,
        *,
        sampler=None,
        total_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _test_loader_from_config(cfg):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = build_dataset_train(cfg)
    _log_api_usage("dataset." + cfg.TEST_DATASET.NAME)

    return {
        "dataset": dataset,
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": cfg.SOLVER.DATA_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_test_loader_from_config)
def build_test_loader(
        dataset: Union[List[Any], torchdata.Dataset],
        *,
        sampler: Optional[torchdata.Sampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn=collate_fn,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return build_batch_data_loader(
        dataset,
        sampler,
        batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
