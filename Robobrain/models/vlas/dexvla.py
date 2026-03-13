import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from functools import partial
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

import torch
import torch.nn as nn

from Robobrain.models.policy.transformer_utils import MAPBlock
from Robobrain.models.vlms import PrismaticVLM
from Robobrain.models.vlms import QwenVLVLM
from Robobrain.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from Robobrain.models.backbones.llm import LLMBackbone
from Robobrain.models.backbones.llm.prompting import PromptBuilder
from Robobrain.models.backbones.vision import VisionBackbone
from Robobrain.overwatch import initialize_overwatch
from Robobrain.models.policy_heads.models.unet_diffusion.modeling_unet_diffusion import ConditionalUnet1D
from Robobrain.models.vlms.qwen_vl import _DYN_TOKEN_MIN, _DYN_TOKEN_MAX, _DEX_TOKEN_MIN, _DEX_TOKEN_MAX

from transformers import AutoConfig, AutoModel

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 5, hidden_dim = 512, vis_dim = 4096):
        super().__init__()
        self.identifier = "MAPBlock Decoder"
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(48, hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                            )

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim * 2, window_size * 48),     # 7-Dof Action Space
                                # nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio.reshape(proprio.shape[0], -1))
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1))
        
        return action

class ActionDecoder_l(torch.nn.Module):
    def __init__(self, window_size = 5, hidden_dim = 512, vis_dim = 4096):
        super().__init__()
        self.identifier = "MAPBlock Decoder"
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(48, hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                    )
        self.res = nn.Sequential(
                                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                nn.ReLU(),
                                nn.LayerNorm(hidden_dim * 2),
                            )

        self.proj = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.Linear(hidden_dim * 2, window_size * 48),
        )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        fused_embed = torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1)
        fused_embed = fused_embed + self.res(fused_embed)
        action = self.proj(fused_embed)
        
        return action

class DexVLA_Model(torch.nn.Module):
    def __init__(self, vlm: QwenVLVLM, use_diffusion_head=True, window_size=12):
        super().__init__()
        self.vlm = vlm
        self.window_size = window_size
        self.use_diffusion_head = use_diffusion_head
        if self.use_diffusion_head :
            self.action_decoder = ConditionalUnet1D(input_dim=48, window_size=self.window_size, vis_dim=2048)
        else:
            self.action_decoder = ActionDecoder(window_size=self.window_size, vis_dim=2048)

        self.all_module_keys = ["vlm.llm_backbone", "action_decoder"]

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
            batch["proprio"] = batch["proprio"].to(next(self.parameters()).device) 
            batch["actions"] = batch["actions"].to(next(self.parameters()).device)
        loss, loss_one_step, latent_action_tokens, loss_batch = self.action_decoder_forward(batch, vla_output, batch["diff_loss_mask"])

        return vla_output, loss, loss_one_step, latent_action_tokens, loss_batch

    def action_decoder_forward(self, batch, slow_output, diff_loss_mask):
        # Task and action latents
        # visual_embed = slow_output.hidden_states[-1][:, : self.vlm.vision_backbone.dino_featurizer.patch_embed.num_patches ].to(torch.float)
        # latent_tokens = slow_output.hidden_states[-1][:, self.vlm.vision_backbone.dino_featurizer.patch_embed.num_patches : ]
        visual_embed = slow_output.hidden_states[-1][:, : self.vlm.vision_backbone.num_patches].to(torch.float)
        latent_tokens = slow_output.hidden_states[-1][:, self.vlm.vision_backbone.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt >= _DYN_TOKEN_MIN

        action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            m = mask[idx]
            if m.any():
                true_indices = torch.where(m)[0]
                # start_index = true_indices[7] + 2
                start_index = true_indices[0]
                per_sample_action_tokens = per_sample_latent_tokens[start_index:start_index+44, :]
                action_tokens.append(per_sample_action_tokens)
            else:
                per_sample_action_tokens = per_sample_latent_tokens[-44:, :]
                action_tokens.append(per_sample_action_tokens)
        action_tokens = torch.stack(action_tokens).to(torch.float)
        if self.use_diffusion_head:
            loss, loss_one_step, loss_batch = self.action_decoder(actions=batch['actions'], visual_embed=visual_embed, action_tokens=action_tokens, proprio=batch["proprio"], diff_loss_mask=diff_loss_mask)
        else:
            pred_action = self.action_decoder(action_tokens, visual_embed, batch["proprio"]).reshape(-1, self.window_size, 48)
            loss_batch = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
            loss_one_step = loss_batch[:,0].mean()
            loss = (loss_batch.mean(dim=(1, 2)) * diff_loss_mask).mean()

        return loss, loss_one_step, action_tokens, loss_batch
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        freeze_weights: bool = True,
        use_diffusion_head: bool = True,
        window_size: int = 8,
        **kwargs,
    ):
        """Initialize a DexVLA_Model from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = QwenVLVLM(
            model_id
        )
        vla = cls(vlm=vlm, use_diffusion_head=use_diffusion_head, window_size=window_size)
        # special for more special tokens
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<DYN_{i}>' for i in range(16)] + [f'<DEX_{i}>' for i in range(512)]}
        num_new_tokens = vla.vlm.llm_backbone.get_tokenizer().add_special_tokens(special_tokens_dict)
        vla.vlm.llm_backbone.llm.resize_token_embeddings(len(vla.vlm.llm_backbone.get_tokenizer()), pad_to_multiple_of=64)
        if num_new_tokens > 0:
            input_embeddings = vla.vlm.llm_backbone.llm.get_input_embeddings().weight.data
            output_embeddings = vla.vlm.llm_backbone.llm.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "vlm.projector" in model_state_dict and "vlm.llm_backbone" in model_state_dict and "action_decoder" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `vlm.projector` AND `vlm.llm_backbone` AND `action_decoder`!"

        vla.vlm.projector.load_state_dict(model_state_dict["vlm.projector"])
        vla.vlm.llm_backbone.load_state_dict(model_state_dict["vlm.llm_backbone"])
        vla.action_decoder.load_state_dict(model_state_dict["action_decoder"])
        if "vlm.vision_backbone" in model_state_dict.keys():
            vla.vlm.vision_backbone.load_state_dict(model_state_dict["vlm.vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vla.vlm.requires_grad_(False)
            vla.vlm.eval()

        return vla
    
    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vlm.vision_backbone.requires_grad_(False)
            self.vlm.llm_backbone.requires_grad_(False)
            self.vlm.projector.requires_grad_(True)
            self.action_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vlm.projector", "action_decoder"]

            # Update Trackers
            self.vlm.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vlm.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.vlm.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.vlm.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Decoder `{self.action_decoder.identifier}`", ctx_level=1)
            

        elif stage in {"finetune", "vla-train"}:
            self.vlm.vision_backbone.requires_grad_(False)
            self.vlm.llm_backbone.requires_grad_(True)
            self.vlm.projector.requires_grad_(True)
            self.action_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vlm.projector", "vlm.llm_backbone", "action_decoder"]

            # Update Trackers
            self.vlm.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vlm.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.vlm.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.vlm.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Decoder `{self.action_decoder.identifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vlm.vision_backbone.dtype = torch.float32
            self.vlm.vision_backbone.requires_grad_(True)
            self.vlm.llm_backbone.requires_grad_(True)
            self.vlm.projector.requires_grad_(True)
            self.action_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vlm.vision_backbone", "vlm.projector", "vlm.llm_backbone", "action_decoder"]

            # Update Trackers
            self.vlm.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vlm.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.vlm.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.vlm.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Decoder `{self.action_decoder.identifier}`", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vlm.vision_backbone.requires_grad_(False)
            self.vlm.projector.requires_grad_(False)
            self.vlm.llm_backbone.requires_grad_(False)
            self.action_decoder.requires_grad_(True)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vlm.llm_backbone", "action_decoder"]

            # Update Trackers
            self.vlm.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vlm.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.vlm.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.vlm.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Action Decoder `{self.action_decoder.identifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vlm.vision_backbone.dtype = torch.float32
            self.vlm.vision_backbone.requires_grad_(True)
            self.vlm.projector.requires_grad_(True)
            self.vlm.llm_backbone.requires_grad_(False)
            self.action_decoder.requires_grad_(True)

            # Unfreeze final LLM layer
            for module in self.vlm.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vlm.vision_backbone", "vlm.projector", "vlm.llm_backbone", "action_decoder"]

            # Update Trackers
            self.vlm.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vlm.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.vlm.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.vlm.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Action Decoder `{self.action_decoder.identifier}`", ctx_level=1)
            # fmt: on

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)
                
    def get_fsdp_wrapping_policy(self):

        vision_llm_projector_policy = self.vlm.get_fsdp_wrapping_policy()

        action_decoder_policy = partial(
            _module_wrap_policy,
            module_classes={ActionDecoder},
        )

        return partial(
            _or_policy,
            policies=[
                vision_llm_projector_policy,
                action_decoder_policy,
            ],
        )