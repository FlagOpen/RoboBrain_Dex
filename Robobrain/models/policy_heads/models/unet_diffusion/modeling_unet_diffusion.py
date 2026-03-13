"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from .configuration_unet_diffusion import UnetDiffusionPolicyConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoModelForCausalLM
import copy
from Robobrain.models.policy.transformer_utils import MAPBlock
# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self.dtype=dtype

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=self.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                input_dim=44, # action dim
                down_dims=[256, 512, 1024],
                window_size=10,
                noise_samples=1,
                global_cond_dim=2048,
                diffusion_step_embed_dim=256,
                kernel_size=5,
                n_groups=8,
                num_inference_timesteps=10,
                num_train_timesteps=100,
                vis_dim = 4096,
        ):
        """
        input_dim: Dim of actions.
        window_size: Number of action chunk size to predict.
        global_cond_dim: Dim of global conditioning to diffusion step embedding.
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        self.identifier = "Diffusion Decoder"
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.action_dim = input_dim

        self.window_size = window_size
        self.noise_samples = noise_samples

        hidden_dim = global_cond_dim // 2
        self.combine = nn.Sequential(
                                nn.Linear(global_cond_dim, global_cond_dim),     # 7-Dof Action Space
                                # nn.Tanh(),
                    )
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(self.action_dim, hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                            )
        
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            # SinusoidalPosEmb(dsed, torch.bfloat16),
            SinusoidalPosEmb(dsed, torch.float),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])
        
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.num_inference_timesteps = num_inference_timesteps
        # self.proj_to_action = nn.Identity()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,  # 100
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

    def forward(self, actions, visual_embed, action_tokens, proprio, diff_loss_mask):
        """
        Forward pass for the diffusion head.
        :param actions: target actions, shape [B, Ta, D] D:10 = 3+6+1
        :param visual_embed: visual embed from the llava_pythia, as the condition for the diffusion, shape [B,Tokens, D]
        :param action_tokens: action tokens from the llava_pythia, as the condition for the diffusion, shape [B,Tokens, D]
        :param proprio: robot states, shape [B, D]
        :return: loss
        """
        if actions is not None:  # training time
            B = actions.size(0)
            num_noise_samples = self.noise_samples
            # sample noise to add to actions
            noise = torch.randn([num_noise_samples] + list(actions.shape), device=actions.device,
                                dtype=actions.dtype)  # num_noise, B, Ta, D
            # sample a diffusion iteration for each data point 
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=actions.device
            ).long()

            timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]

            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            assert visual_embed.ndim == 3
            assert action_tokens.ndim == 3

            visual_embed = visual_embed.repeat(num_noise_samples, 1, 1)
            action_tokens = action_tokens.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            proprio = proprio.repeat(num_noise_samples, 1)

            noise_pred = self.model_forward(noisy_actions, timesteps, visual_embed=visual_embed, action_tokens=action_tokens, proprio=proprio)
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss_batch = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss_one_step = loss_batch[:,0].mean()
            loss = (loss_batch.mean(dim=(1, 2)) * diff_loss_mask).mean()
            # loss_dict['loss'] = loss
            return loss, loss_one_step, loss_batch
            # return loss
        else:  # inference time
            B = 1
            Tp = self.window_size
            action_dim = self.action_dim

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, Tp, action_dim)).cuda()

            naction = noisy_action.to(dtype=visual_embed.dtype)
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.model_forward(naction, k, visual_embed=visual_embed, action_tokens=action_tokens, proprio=proprio)

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def model_forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                visual_embed: torch.Tensor=None,
                action_tokens: torch.Tensor=None,
                proprio=None):
        """
        sample: (B,T,input_dim), noisy action
        timestep: (B,) or int, diffusion step
        visual_embed: (B,Tokens,hidden_dim)
        action_tokens: (B,Tokens,hidden_dim)
        proprio: (B,action_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        
        global_cond = self.combine(torch.cat([self.attn_pool(action_tokens, init_embed=visual_embed), proprio], dim=-1))
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        assert timesteps.shape[0] == sample.shape[0]

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x

