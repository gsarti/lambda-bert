"""
Model configuration, following HF formalism
"""

import logging
from typing import Callable, Union

from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class LambdaBertConfig(PretrainedConfig):
    model_type = "lambdabert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        classifier_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-12,
        pad_token_id=0,
        gradient_checkpointing: bool = False,
        num_lambda_queries: int = 8,
        key_depth: int = 8,
        intra_depth: int = 4,
        local_context_size: int = None,
        output_lambda_params: bool = True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_epsilon
        self.gradient_checkpointing = gradient_checkpointing

        # LambdaNetworks-specific parameters

        # dim_v
        self.num_lambda_queries = num_lambda_queries

        # dim_k
        self.key_depth = key_depth

        # dim_u
        self.intra_depth = intra_depth

        # n
        self.local_context_size = local_context_size

        # If true, outputs λc instead of attention weights
        self.output_lambda_params = output_lambda_params
