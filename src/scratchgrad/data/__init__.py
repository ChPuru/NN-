"""Datasets, loaders, and hand-written parsers for MNIST, CIFAR-10 and text."""

from __future__ import annotations

from .cifar10 import CIFAR10_CLASSES, cifar10, load_cifar10, unpack_rows
from .dataset import (
    DataLoader, Dataset, Subset, TensorDataset, default_collate, random_split,
)
from .mnist import (
    fashion_mnist, load_fashion_mnist, load_mnist, mnist,
    parse_idx_images, parse_idx_labels, to_float,
)
from .text import CharDataset, CharVocab, load_text, normalize_text, tiny_shakespeare

__all__ = [
    "Dataset", "TensorDataset", "Subset", "random_split", "default_collate", "DataLoader",
    "mnist", "load_mnist", "fashion_mnist", "load_fashion_mnist",
    "parse_idx_images", "parse_idx_labels", "to_float",
    "cifar10", "load_cifar10", "unpack_rows", "CIFAR10_CLASSES",
    "CharVocab", "CharDataset", "load_text", "normalize_text", "tiny_shakespeare",
]
