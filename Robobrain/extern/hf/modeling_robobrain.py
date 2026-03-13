"""
modeling_robobrain.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import types
from contextlib import contextmanager

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from Robobrain.models.policy.transformer_utils import MAPBlock
from Robobrain.models.policy_heads.models.unet_diffusion.modeling_unet_diffusion import ConditionalUnet1D
from .configuration_robobrain import OpenVLAConfig, PrismaticConfig, DexVLAConfig
from Robobrain.models.action_heads import TransformerHead
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationMixin,
)
from Robobrain.models.vlms.qwen_vl import _DYN_TOKEN_MIN, _DYN_TOKEN_MAX, _DEX_TOKEN_MIN, _DEX_TOKEN_MAX
from transformers.modeling_outputs import CausalLMOutputWithPast

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma

# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings
    
    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        use_proprio=False,
        proprio_projector=None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            if use_proprio:
                projected_patch_embeddings = self._process_proprio_features(
                    projected_patch_embeddings, proprio, proprio_projector
                )
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions


    def predict_latent_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        output = self.generate(input_ids, min_new_tokens=30, max_new_tokens=30, return_dict_in_generate=True, output_hidden_states=True, **kwargs)
        generated_ids = output.sequences

        
        last_hidden_states = [hidden_states[-1] for hidden_states in output.hidden_states]
        latent_tokens = torch.cat(last_hidden_states, dim=1)#[:, :-1]
        visual_embed = latent_tokens[:,:256]
        latent_tokens = latent_tokens[:, 256:]

        # print(generated_ids)
        latent_mask = generated_ids > 32002
        latent_mask = latent_mask[:, 1:]
        # print(latent_mask[0])
        # latent_action = latent_tokens[:, latent_mask[0], :]
        latent_action = latent_tokens[:, -4:]
        generated_ids = generated_ids[:, 1:][:, latent_mask[0]]
        generated_ids = generated_ids[:, -4:]

        return latent_action, visual_embed, generated_ids


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 5, input_dim=48, hidden_dim = 512, vis_dim = 4096):
        super().__init__()
        self.identifier = "MAPBlock Decoder"
        self.window_size = window_size
        self.input_dim = input_dim
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(input_dim , hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                            )

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim * 2, window_size * input_dim),     # 7-Dof Action Space
                                # nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio.reshape(proprio.shape[0], -1))
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1))
        
        return action
    
class ActionDecoder_l(torch.nn.Module):
    def __init__(self, window_size = 5, input_dim=48, hidden_dim = 512, vis_dim = 4096):
        super().__init__()
        self.identifier = "MAPBlock Decoder"
        self.input_dim = input_dim
        self.window_size = window_size
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                    )
        self.res = nn.Sequential(
                                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                nn.ReLU(),
                                nn.LayerNorm(hidden_dim * 2),
                            )

        self.proj = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.Linear(hidden_dim * 2, window_size * input_dim),
        )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio.reshape(proprio.shape[0], -1))
        visual_embed = self.visual_pool(visual_embed)
        fused_embed = torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1)
        fused_embed = fused_embed + self.res(fused_embed)
        action = self.proj(fused_embed)
        
        return action

class ActionDecoder_noproprio(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512, input_dim=48, vis_dim = 4096):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, self.input_dim * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action

class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

class DexVLAForActionPrediction(Qwen2_5_VLForConditionalGeneration):
    config_class: PretrainedConfig = DexVLAConfig

    def __init__(self, config: DexVLAConfig) -> None:
        super().__init__(config)
        self.use_diffusion_head = config.use_diffusion_head
        self.use_flow_matching = config.use_flow_matching
        self.window_size = config.window_size
        self.input_dim = config.input_dim
        self.use_proprio = config.use_proprio
        self.llm_dim = config.text_config.hidden_size
        if config.use_diffusion_head :
            self.action_decoder = ConditionalUnet1D(input_dim=self.input_dim, window_size=config.window_size, vis_dim=2048)
        elif config.use_flow_matching:
            self.action_decoder = FlowMatchingActionDecoder(input_dim=self.input_dim, window_size=config.window_size, vis_dim=2048)
        else:
            self.action_decoder = ActionDecoder(input_dim=48, window_size=config.window_size, vis_dim=2048, hidden_dim=512)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        self.num_patches = 256
        if self.use_proprio:
            self.num_patches += 1
        
        self.motion_dynamics = None

    def vlm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:    

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if image_grid_thw is None and "image_grid_thw" in kwargs:
            image_grid_thw = kwargs["image_grid_thw"]
        
        if "position_ids" not in kwargs and input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            kwargs["position_ids"] = position_ids
       
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            **kwargs
        )
        
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        
        return model_inputs

    def predict_latent_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, proprio = None, num_steps=10, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 198):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([198]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        output = self.generate(input_ids, min_new_tokens=50, max_new_tokens=50, return_dict_in_generate=True, output_hidden_states=True, **kwargs)
        generated_ids = output.sequences

        
        last_hidden_states = [hidden_states[-1] for hidden_states in output.hidden_states]
        latent_tokens = torch.cat(last_hidden_states, dim=1)#[:, :-1]

        visual_embed = latent_tokens[:,:self.num_patches].to(torch.float)
        latent_tokens = latent_tokens[:, self.num_patches:]

        latent_mask = generated_ids[0] >= _DYN_TOKEN_MIN
        
        if latent_mask.any():
            true_indices = torch.where(latent_mask)[0]

            start_index = true_indices[0]
            per_sample_action_tokens = latent_tokens[0][start_index:start_index+44, :]
            action_tokens = per_sample_action_tokens.unsqueeze(0).to(torch.float)
            self.motion_dynamics = action_tokens.cpu().numpy()
            if self.use_flow_matching:
                device = proprio.device
                actions_shape = (1, self.window_size, self.input_dim)
                noise = self.sample_noise(actions_shape, device)
                dt = -1.0 / num_steps
                dt = torch.tensor(dt, dtype=torch.float32, device=device)
                x_t = noise
                time = torch.tensor(1.0, dtype=torch.float32, device=device)
                while time >= -dt / 2:
                    expanded_time = time.expand(1)
                    v_t = self.action_decoder(
                        action_tokens, 
                        visual_embed, 
                        proprio,
                        x_t,
                        expanded_time,
                    )
                    x_t = x_t + dt * v_t
                    time += dt
                pred_action = x_t
            else:
                pred_action = self.action_decoder(action_tokens, visual_embed, proprio).reshape(-1, self.window_size, self.input_dim)
        return pred_action, visual_embed, generated_ids

    def forward(self, *args, **kwargs):
        ## Training Mode
        if args and isinstance(args[0], dict): 
            batch = args[0]
            if self.use_proprio:
                proprio_projector = args[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vla_output = self.vlm_forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    labels=batch["labels"],
                    output_hidden_states=True,        # Return intermediate tokens of all layers
                    proprio=batch["proprio"],
                    use_proprio=self.use_proprio,
                    proprio_projector=proprio_projector if self.use_proprio else None,
                )
            batch["proprio"] = batch["proprio"].to(next(self.parameters()).device) 
            batch["actions"] = batch["actions"].to(next(self.parameters()).device)
            loss, loss_one_step, latent_action_tokens, loss_batch = self.action_decoder_forward(batch, vla_output, batch["diff_loss_mask"])

            return vla_output, loss, loss_one_step, latent_action_tokens, loss_batch
        ## Inference Mode
        else:
            return super(self.__class__, self).forward(*args, **kwargs)
        
    def action_decoder_forward(self, batch, slow_output, diff_loss_mask):
        # Task and action latents
        visual_embed = slow_output.hidden_states[-1][:, : self.num_patches].to(torch.float)
        latent_tokens = slow_output.hidden_states[-1][:, self.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt >= _DYN_TOKEN_MIN

        action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            m = mask[idx]
            if m.any():
                true_indices = torch.where(m)[0]

                start_index = true_indices[0]
                per_sample_action_tokens = per_sample_latent_tokens[start_index:start_index+44, :]
                action_tokens.append(per_sample_action_tokens)
            else:
                per_sample_action_tokens = per_sample_latent_tokens[-44:, :]
                action_tokens.append(per_sample_action_tokens)
        action_tokens = torch.stack(action_tokens).to(torch.float)
        if self.use_diffusion_head:
            loss, loss_one_step, loss_batch = self.action_decoder(actions=batch['actions'], visual_embed=visual_embed, action_tokens=action_tokens, proprio=batch["proprio"], diff_loss_mask=diff_loss_mask)
        elif self.use_flow_matching:
            actions = batch['actions'].to(torch.float)
            noise = self.sample_noise(actions.shape, actions.device)
            time = self.sample_time(actions.shape[0], actions.device)
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            v_pred = self.action_decoder(action_tokens, visual_embed, batch["proprio"], x_t, time).reshape(u_t.shape[0], self.window_size, self.input_dim)

            loss_batch = F.mse_loss(u_t, v_pred, reduction='none')
            loss_one_step = loss_batch[:,0].mean()
            loss = (loss_batch.mean(dim=(1, 2)) * diff_loss_mask).mean()
        else:
            pred_action = self.action_decoder(action_tokens, visual_embed, batch["proprio"]).reshape(-1, self.window_size, self.input_dim)
            loss_batch = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
            loss_one_step = loss_batch[:,0].mean()
            loss = (loss_batch.mean(dim=(1, 2)) * diff_loss_mask).mean()


        return loss, loss_one_step, action_tokens, loss_batch
    
    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    
    def get_proprio_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["proprio"]

    # Flow Matching head
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
    
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float32, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

class ResidualMLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner), 
            nn.GELU(),
            nn.Linear(inner, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

# exp3
class FMBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # x: (B,W,d), context: (B,Tctx,d), gamma/beta: (B,d)
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        h = self.ln2(x)
        x = x + self.cross_attn(h, context, context, need_weights=False)[0]

        h = self.ln3(x)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)   
        x = x + self.mlp(h)
        return x

class FlowMatchingActionDecoder(nn.Module):
    def __init__(
        self,
        window_size: int = 16, input_dim: int = 48,
        hidden_dim: int = 512,   
        d_model: int = 768,      
        n_layers: int = 6, n_heads: int = 12,
        time_dim: int = 256, vis_dim: int = 4096, dropout: float = 0.1,
    ):
        super().__init__()
        self.identifier = "FMTransformer(MAP)"
        self.window_size = window_size
        self.input_dim = input_dim
        self.d_model = d_model

        self.visual_pool = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.attn_pool   = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.cond_proj = nn.Linear(hidden_dim, d_model)

        self.act_in  = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, window_size, d_model) / math.sqrt(d_model))

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2)   # [gamma, beta]
        )

        self.blocks = nn.ModuleList([
            FMBlock(d_model, n_heads, mlp_mult=4, dropout=dropout) for _ in range(n_layers)
        ])
        self.out  = nn.Linear(d_model, input_dim)
        self.skip = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.zeros_(self.skip.weight)

        self.cond_fuse = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU())

        self.time_dim = time_dim

    def _make_context(self, latent_action_tokens: Tensor, visual_embed: Tensor, proprio: Tensor) -> Tensor:
        visual_token = self.visual_pool(visual_embed)            # (B, H) or (B,1,H)
        if visual_token.dim() == 3 and visual_token.size(1) == 1:
            visual_token = visual_token.squeeze(1)

        attn_h = self.attn_pool(latent_action_tokens, init_embed=visual_token) # (B,H) or (B,1,H)
        if attn_h.dim() == 3 and attn_h.size(1) == 1:
            attn_h = attn_h.squeeze(1)

        proprio_h = self.proprio_proj(proprio.reshape(proprio.shape[0], self.input_dim))                   # (B, H)

        v = self.cond_proj(visual_token)  # (B,d)
        a = self.cond_proj(attn_h)        # (B,d)
        p = self.cond_proj(proprio_h)     # (B,d)

        # context: (B,3,d)
        ctx = torch.stack([a, v, p], dim=1)
        return ctx

    def forward(self, latent_action_tokens, visual_embed, proprio, a_t, timestep) -> Tensor:
        B, W, D = a_t.shape
        assert W == self.window_size and D == self.input_dim

        context = self._make_context(latent_action_tokens, visual_embed, proprio)   # (B,3,d_model)
        x = self.act_in(a_t) + self.pos_emb[:, :W, :]   # (B,W,d_model)
        t_emb = create_sinusoidal_pos_embedding(timestep, self.time_dim, min_period=4e-3, max_period=4.0, device=timestep.device)     # (B,time_dim)
        gamma, beta = self.time_mlp(t_emb).chunk(2, dim=-1)      # (B,d_model), (B,d_model)

        for blk in self.blocks:
            x = blk(x, context, gamma, beta)   # (B,W,d_model)

        v_pred = self.out(x)                   # (B,W,D)
        return v_pred