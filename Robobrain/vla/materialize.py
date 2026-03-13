"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from Robobrain.models.backbones.llm.prompting import PromptBuilder
from Robobrain.models.backbones.vision import ImageTransform
from Robobrain.util.data_utils import PaddedCollatorForActionPrediction
from Robobrain.vla.action_tokenizer import ActionTokenizer
from Robobrain.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSQwenVLBatchTransform, RLDSBatchTransformDex_withHis,  RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator


def get_cot_latent_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    image_transform_lam: ImageTransform,
    latent_action_tokenizer: PreTrainedTokenizerBase, 
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    window_size: int = 5,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # action_tokenizer = ActionTokenizer(tokenizer)

    batch_transform = RLDSQwenVLBatchTransform(
        action_tokenizer=latent_action_tokenizer,
        qwen_processor=tokenizer,
        image_transform=image_transform,
        image_transform_lam=image_transform_lam,
        prompt_builder_fn=prompt_builder_fn,
        window_size=window_size
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.tokenizer.model_max_length, tokenizer.tokenizer.pad_token_id, padding_side=padding_side, num_padding=0
    )


    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=window_size + 1, 
        training_phase='pre-training',
    )

    return dataset, tokenizer.tokenizer, collator

def get_subtask_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    image_transform_lam: ImageTransform,
    latent_action_tokenizer: PreTrainedTokenizerBase, 
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # action_tokenizer = ActionTokenizer(tokenizer)

    batch_transform = RLDSBatchTransformDex_withHis(
        action_tokenizer=latent_action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        image_transform_lam=image_transform_lam,
        prompt_builder_fn=prompt_builder_fn
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )


    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        training_phase='pre-training',
    )

    return dataset, tokenizer, collator