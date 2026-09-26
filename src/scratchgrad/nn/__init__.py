"""Layers, losses and initialization."""

from __future__ import annotations

from . import functional, init
from .attention import (
    GPT,
    MLP,
    MultiHeadAttention,
    PositionalEmbedding,
    TransformerBlock,
    causal_mask,
    mask_value,
    scaled_dot_product_attention,
)
from .conv import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    Conv1d,
    Conv2d,
    MaxPool2d,
    ZeroPad2d,
    adaptive_avg_pool2d,
    avg_pool2d,
    col2im,
    conv1d,
    conv2d,
    conv_output_size,
    im2col,
    max_pool2d,
    zero_pad2d,
)
from .layers import (
    ELU,
    GELU,
    Flatten,
    Identity,
    Lambda,
    LeakyReLU,
    Linear,
    LogSoftmax,
    ReLU,
    Reshape,
    Sigmoid,
    SiLU,
    Softmax,
    Softplus,
    Tanh,
)
from .loss import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
    NLLLoss,
)
from .module import Module, ModuleDict, ModuleList, Parameter, Sequential
from .norm import (
    BatchNorm1d,
    BatchNorm2d,
    Dropout,
    Dropout2d,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    dropout2d,
    layer_norm,
    rms_norm,
)
from .rnn import GRU, LSTM, RNN, Embedding, GRUCell, LSTMCell, RNNCell, detach_state, embedding
from .scan import MinGRU, SelectiveSSM, linear_scan, selective_scan
from .vae import VAE, gaussian_kl, negative_elbo, reparameterize

__all__ = [  # noqa: RUF022 -- grouped by category, which is more useful here than alphabetical
    # containers
    "Module",
    "Parameter",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    # dense and structural
    "Linear",
    "Identity",
    "Flatten",
    "Reshape",
    "Lambda",
    # activations
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "Softplus",
    "Softmax",
    "LogSoftmax",
    # convolution
    "Conv1d",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "ZeroPad2d",
    "im2col",
    "col2im",
    "conv_output_size",
    "conv1d",
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "zero_pad2d",
    # normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "GroupNorm",
    "RMSNorm",
    "Dropout",
    "Dropout2d",
    "layer_norm",
    "rms_norm",
    "dropout2d",
    # sequences
    "Embedding",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    "LSTM",
    "GRU",
    "detach_state",
    "embedding",
    "linear_scan",
    "selective_scan",
    "MinGRU",
    "SelectiveSSM",
    # attention
    "scaled_dot_product_attention",
    "causal_mask",
    "mask_value",
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "PositionalEmbedding",
    "GPT",
    # generative
    "VAE",
    "gaussian_kl",
    "negative_elbo",
    "reparameterize",
    # losses
    "CrossEntropyLoss",
    "NLLLoss",
    "MSELoss",
    "L1Loss",
    "HuberLoss",
    "BCEWithLogitsLoss",
    # submodules
    "functional",
    "init",
]
