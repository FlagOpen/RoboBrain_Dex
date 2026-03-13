### For pose-only experiments ###

import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
from thop import profile
from fvcore.nn.flop_count import flop_count
from tqdm import tqdm
import time


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_dynamics_model.genie.modules.hand_encoder.utils import count_params
from motion_dynamics_model.genie.modules.hand_encoder.microaction_encoder import MicroactionEncoder
from motion_dynamics_model.genie.modules.hand_encoder.ms_tcn_1D import MultiScale_TemporalConv as MS_TCN
from motion_dynamics_model.genie.modules.hand_encoder.transformer_unimodal import Unimodal_TF


# Notations
# N: batch size
# C: 2 or 3 (channel dimension/#coordinates)
# T: #frames
# J: #joints
# E: #entities (2 if hands are separated)


def normalize_tensor(tensor):
    """
    Normalize a tensor so that the second dimension (C) has unit length for each entry.
    Args:
    tensor (torch.Tensor): A tensor of shape (N, C, T, M).
    Returns:
    torch.Tensor: Normalized tensor.
    """
    # Calculate the norm of the second dimension (C)
    norms = torch.norm(tensor, dim=1, keepdim=True)
    norms = norms.clamp(min=1e-12) # Avoid division by zero
    normalized_tensor = tensor / norms # Normalize the tensor
    return normalized_tensor


def quaternion_to_rotation_vector(quat):

    quat_norm = torch.norm(quat, dim=1, keepdim=True)
    quat = quat / (quat_norm + 1e-8)
    
    w = quat[:, 0:1, :, :]  # (N, 1, T, 1, 2)
    x = quat[:, 1:2, :, :]  # (N, 1, T, 1, 2)
    y = quat[:, 2:3, :, :]  # (N, 1, T, 1, 2)
    z = quat[:, 3:4, :, :]  # (N, 1, T, 1, 2)

    angle = 2 * torch.acos(torch.clamp(w, -1 + 1e-8, 1 - 1e-8))
    
    sin_half_angle = torch.sqrt(torch.clamp(1 - w**2, 0, 1))

    epsilon = 1e-8
    sin_half_angle = torch.where(sin_half_angle < epsilon, epsilon, sin_half_angle)
  
    axis_x = x / sin_half_angle
    axis_y = y / sin_half_angle
    axis_z = z / sin_half_angle

    small_angle = angle < epsilon
    axis_x = torch.where(small_angle, torch.zeros_like(axis_x), axis_x)
    axis_y = torch.where(small_angle, torch.zeros_like(axis_y), axis_y)
    axis_z = torch.where(small_angle, torch.zeros_like(axis_z), axis_z)
    
    rot_vec = torch.cat([axis_x * angle, axis_y * angle, axis_z * angle], dim=1)
    
    return rot_vec

class HF_Pose(nn.Module):
    def __init__(self, microaction_window_size, num_joints, 
                 embedding_dim_final=256, use_2d_pose=False, dropout=0, microaction_overlap=0.0, # [0.0, 0.99)
                 trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=3, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                 use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                 tf_heads=8, tf_layers=2):
        super().__init__()

        """
        Args:
        microaction_window_size (int): Number of frames in a microaction.
        num_joints (int): Number of joints in the skeleton.
        # num_classes (int): Number of classes in the dataset.
        embedding_dim_final (int): The target dimension for microaction encoding at the end of three layers. Also transformer input dimension.
        use_2d_pose (bool): If True, 2D pose is used. Otherwise, 3D pose is used.
        dropout (float): Dropout rate.
        microaction_overlap (float): Overlap between consecutive microactions. Usual values: 0.0, 0.5, 0.67.
        trajectory_atten_dim_per_head (int): Dimension per head in the trajectory self-attention.
        trajectory_tcn_kernel_size (int): Kernel size of the trajectory TCNs. Usual values: 3, 5.
        trajectory_tcn_stride (list): Strides of the trajectory TCNs. Three values for three layers.
        trajectory_tcn_dilations (list): Dilations of the trajectory TCNs for multi-scale. Same tuple for all layers.
        use_global_wrist_reference (bool): If True, global wrist reference is used.
        include_orientation_in_global_wrist_ref (bool): If True, hand orientation is included in the global wrist reference.
        use_both_wrists (bool): If True, both wrists are used.
        separate_hands (bool): If True, shared pose encoder is applied on each hand separately, then aggregated.
        tf_heads (int): Number of heads in the transformer.
        tf_layers (int): Number of layers in the transformer.
        """

        self.seg_len = microaction_window_size
        self.num_joints = num_joints
        self.use_2d_pose = use_2d_pose
        self.use_global_wrist_reference = use_global_wrist_reference
        self.include_orientation_in_global_wrist_ref = include_orientation_in_global_wrist_ref
        self.use_both_wrists = use_both_wrists
        self.separate_hands = separate_hands # If true, shared pose encoder applied on each hand separately, then aggregated.


        # progressively increasing the dimensions for microaction encoder and wrist TCN
        embedding_dims = [embedding_dim_final // i for i in [4, 2, 1]] # For 256 --> [64, 128, 256]
        layerwise_num_heads = [dim // trajectory_atten_dim_per_head for dim in embedding_dims] # For 256 --> [64//4, 128//4, 256//4] = [16, 32, 64]        
        coordinate_dim = 2 if self.use_2d_pose else 3 # 2D or 3D pose
        self.mactions_in_window = round(1.0/(1.0-microaction_overlap))

        
        self.pose_enc = MicroactionEncoder(coordinate_dim, embedding_dim=embedding_dim_final, num_heads=layerwise_num_heads, dropout=dropout, \
                                            num_frames=microaction_window_size, num_joints=num_joints, num_hands=1 if self.separate_hands else 2, \
                                            stride=trajectory_tcn_stride, kernel_size=trajectory_tcn_kernel_size, dilations=trajectory_tcn_dilations,\
                                            global_wrist_ref=self.use_global_wrist_reference)
        
        if self.use_global_wrist_reference:
            wirst_tcn_input_dim = coordinate_dim * (2 if self.use_both_wrists else 1) # double if both hands are used
            wirst_tcn_input_dim = wirst_tcn_input_dim * (3 if self.include_orientation_in_global_wrist_ref else 1) # double if orientations are included
            # wirst_tcn_input_dim = wirst_tcn_input_dim * (2 if self.include_orientation_in_global_wrist_ref else 1) # double if orientations are included
            self.wrist_data_bn = nn.BatchNorm1d(wirst_tcn_input_dim)
            self.wrist_tcn = nn.Sequential(
                MS_TCN(wirst_tcn_input_dim, embedding_dims[0], stride=trajectory_tcn_stride[0]),
                MS_TCN(embedding_dims[0], embedding_dims[0]),
                MS_TCN(embedding_dims[0], embedding_dims[0]),
                MS_TCN(embedding_dims[0], embedding_dims[1], stride=trajectory_tcn_stride[1]),
                MS_TCN(embedding_dims[1], embedding_dims[1]),
                MS_TCN(embedding_dims[1], embedding_dims[1]),
                MS_TCN(embedding_dims[1], embedding_dims[2], stride=trajectory_tcn_stride[2]),
                MS_TCN(embedding_dims[2], embedding_dims[2]),
                MS_TCN(embedding_dims[2], embedding_dims[2])
            )

            
    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_dict_original = torch.load(pretrained_weights_path)
        model_dict = self.state_dict()
        # Filter out mismatched weights
        pretrained_dict = {k: v for k, v in pretrained_dict_original.items() if k in model_dict and v.size() == model_dict[k].size()}
        # # Load the remaining weights
        model_dict.update(pretrained_dict)
        return model_dict

    def forward(self, x, wrist, x_rgb=None, return_atten_map=False):
        # x: N, C, T, V, M
     
        if self.use_global_wrist_reference:
            if self.include_orientation_in_global_wrist_ref:
                if self.use_both_wrists:
        
                    xw = wrist[:, 0:3, :, 0, :]  
                    wrist_quat = wrist[:, 3:9, :, 0, :]  
                else:
                    # For single hand, we only take the right hand wrist (as left hand is normalized by xf transformation).
                    xw = wrist[:, 0:3, :, 0, 1].unsqueeze(-1) 
                    wrist_quat = wrist[:, 3:9, :, 0, 1].unsqueeze(-1) 
                 
                xw = torch.cat([xw, wrist_quat], dim=1)
            else:
                xw = wrist[:, 0:3, :, 0, 1] if self.use_both_wrists else wrist[:, 0:3, :, 0, 1] .unsqueeze(-1)
            
            wrist_feat = xw.permute(0,3,1,2).contiguous() # (N, C, T, M) to (N, M, C, T)

            N, n_wrist, n_chan, T = wrist_feat.shape
            wrist_feat = wrist_feat.view(N, n_wrist*n_chan, T) # (B, 6, 120)
            wrist_feat = self.wrist_data_bn(wrist_feat)
            wrist_feat = self.wrist_tcn(wrist_feat)
            wrist_feat_token = wrist_feat.mean(dim=-1) # Temporal pooling
        
        x = x[:, :, :, 0:self.num_joints, :] # With default indexing; 6: Fingertips+wrist, 11: include all joints from index & thumb

        N, C, T, V, M = x.shape
        x = x.permute(0,1,3,4,2).contiguous() # Becomes (N, C, V, M, T)
        x = x.unfold(-1, self.seg_len, self.seg_len) if self.mactions_in_window == 1 else x.unfold(-1, self.seg_len, (self.seg_len+1)//self.mactions_in_window)
        x = x.permute(0,4,5,1,2,3).contiguous() # Becomes (N, #microactions, microaction_window_size, C, V, M)
        num_mactions = x.shape[1] # Number of microactions. 8 for 120 frames and 15 frames per microaction when overlap is 0.0.
        x = x.view(N*num_mactions, self.seg_len, C, V, M)
        x = x.permute(0,2,1,3,4).contiguous() # e.g., (N*8, C, 15, V, M)

        if self.separate_hands:
            N_, C_, T_, V_, M_ = x.shape
            x = x.permute(0,4,1,2,3).contiguous().view(N_*M_, C_, T_, V_).unsqueeze(-1) # Becomes N, M, C, T, V --> N*M, C, T, V, 1
            out_all = self.pose_enc(x, global_wrist_token = wrist_feat_token if self.use_global_wrist_reference else None)
            out_all = out_all.view(N_, M_, *out_all.shape[1:])
            N0, M0, C0, J0 = out_all.shape
            out_all = out_all.reshape(N0, C0, M0*J0)
        else:
            out_all = self.pose_enc(x, global_wrist_token = wrist_feat_token if self.use_global_wrist_reference else None) # Becomes (N*8, 256)

        # out_all = out_all.view(N, num_mactions, -1)
        out_all = out_all.view(N, num_mactions, *out_all.shape[1:])

        
     
        out = out_all.permute(0, 1, 3, 2)
    
        return out
