from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from motion_dynamics_model.genie.modules.blocks import SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer
import os
from motion_dynamics_model.genie.modules.hand_encoder.hf_pose import HF_Pose
from motion_dynamics_model.genie.modules.hand_encoder.pose_decoder import MicroactionDecoderTCN
from motion_dynamics_model.genie.modules.rq import ResidualVQ
from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RQ_HandLatentActionModel(nn.Module):
    """
    video and hand Latent action VQ-VAE.
    video and hand latent action from DIFFERENT codebooks
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int, # 48
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(RQ_HandLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4 #seperate for video and hand action
        self.time_steps = 2
        self.action_latent_video = nn.Parameter(torch.empty(1, self.time_steps, self.num_codes, dino_dim))  # TODO: num of codes
        nn.init.uniform_(self.action_latent_video, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook_video = nn.Linear(model_dim, latent_dim)
        self.to_codebook_hand = nn.Linear(model_dim, latent_dim)

        self.vq_video = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )

        self.vq_hand = ResidualVQ(
            dim=latent_dim,                  
            num_quantizers=2,       
            codebook_size=512,      
            shared_codebook=True,       
            kmeans_init=True,
            kmeans_iters=50,
            threshold_ema_dead_code=4,  
            use_cosine_sim=True,
        )

        # hand action projector
        self.num_joints=10 #10 or 12; 10 finger tips+ 2 wrist

        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.hand_up=nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim, # Dim of hand action
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.num_finger=5
        self.D_per_finger=3
        self.num_hand=2

        self.action_encoder = HF_Pose(microaction_window_size=8, num_joints=5, 
                    embedding_dim_final=768, use_2d_pose=False, dropout=0.0, 
                    trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=5, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                    use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                    tf_heads=8, tf_layers=2
        ).cuda()

        self.action_decoder = MicroactionDecoderTCN(
            num_frames=8, num_joints=5, num_hands=2,
            embedding_dim=768, coord_dim=3, width=256, num_upsamples=3
            )

    def vq_encode(self, videos: Tensor, actions: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)
        
        B, T, action_dim = actions.shape
        if T < 16:
            raise ValueError(f"actions must have at least 16 frames, but got {T}")
        elif T > 16:
            idx = torch.linspace(0, T - 1, steps=16, device=actions.device)
            idx = torch.round(idx).clamp(0, T - 1).long()
            action = actions.index_select(1, idx)
        else:
            action = actions
       
        wrist_action = action[:, :, :18] 
        finger_action = action[:, :, 18:] 
        finger_action = finger_action.reshape(B, action.shape[1], self.num_finger, self.D_per_finger, self.num_hand) 
        finger_action = finger_action.permute(0, 3, 1, 2, 4)
 
        wrist_action = wrist_action.reshape(B, action.shape[1], 9, 2)
        wrist_action = wrist_action.permute(0, 2, 1, 3)  # [B, 9, T, 2]
        wrist_action = wrist_action.unsqueeze(3)  # [B, 9, T, 1, 2]
        action_feature = self.action_encoder(finger_action, wrist_action)   # [B , 2, 10, 768]

        # video token
        B_a, T_a, L_a, D_a = action_feature.shape
        action_pad_video = self.action_latent_video.expand(B, T_a, -1, -1)
        padded_patches_video = torch.cat([action_pad_video, dion_features, action_feature], dim=2) 
        # Encode
        z = self.encoder(padded_patches_video) 
        latent_feature = z[:, :, :self.num_codes]

        action_feat = z[:, :, -L_a:]  # (B, T, L_action, D)
        
        z_video = self.to_codebook_video(latent_feature[:,1:,:self.num_codes])  # (B, T-1, n, E)
        z_video = z_video.reshape(B , (self.time_steps -1) * self.num_codes, self.latent_dim)
        z_q_video, z_video, emb_video, indices_video = self.vq_video(z_video) #quantenized latent action of video 
        z_q_video = z_q_video.reshape(B, self.time_steps - 1, self.num_codes, self.latent_dim)
 
        # #hand token
        z_action = action_feat # B 2 10 768
        z_action = self.to_codebook_hand(z_action)
        B_a, T_a, L_a, D_a = z_action.shape # B 2 10 128
        z_hand_feature = z_action.reshape(B_a, T_a * L_a, D_a) # B 2 *10 768

        z_q_hand,  indices_hand, losses_hand_commit = self.vq_hand(z_hand_feature) #quantenized latent action of hand 
        indices_hand = indices_hand.reshape(B_a, -1)
        z_q_hand = z_q_hand.reshape(B_a, T_a, L_a, D_a)
        z_q_hand_up = self.hand_up(z_q_hand)
        recon_hand = self.action_decoder(z_q_hand_up)  # (B, M, C, T, J, E)
        recon_hand = recon_hand.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, M, T, C, J, E) 32 2 8 3 5 2
        B, M, T, C, J, E = recon_hand.shape  # 32, 2, 3, 8, 5, 2
        recon_hand = recon_hand.reshape(B, M * T, C * J * E)  # (2, 3*8, 8*5*2)

        return {
            "video_features": dion_features,
            "action_features":action_feature,
            "action_target":action[:, :, 18:] , #raw_actions
            "z_q_video": z_q_video,
            "z_video":z_video,
            "emb_video":emb_video,
            "indices_video":indices_video,
            "action_recon":recon_hand,
            "indices_hand":indices_hand,
            "losses_hand_commit":losses_hand_commit,
        }

    #=========unified video and uni token training=====================
    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        action = batch["action"]
        outputs = self.vq_encode(batch["videos"], action)
        dino_patches = self.patch_up(outputs["video_features"][:, :-1])
        video_patches = self.action_up(outputs["z_q_video"])
        video_uni_patches = torch.cat([video_patches, dino_patches], dim=2)
        
        recon = self.decoder(video_uni_patches)
        video_recon=recon[:, :, self.num_codes * (self.time_steps - 1):]

        outputs.update(
            {
                "video_recon": video_recon,
                "video_target": outputs["video_features"][:, [-1]],
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device     
 


