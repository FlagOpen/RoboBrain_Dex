import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act2 = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + x
        out = self.act2(out)
        return out

class MicroactionDecoderTCN(nn.Module):
    def __init__(
        self,
        num_frames: int,         
        num_joints: int,
        num_hands: int,
        embedding_dim: int = 256, 
        coord_dim: int = 3,    
        width: int = 256,
        num_upsamples: int = 3,  
        kernel_size: int = 3,
        use_final_interp: bool = True
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_hands = num_hands
        self.embedding_dim = embedding_dim
        self.coord_dim = coord_dim

        self.proj_in = nn.Linear(embedding_dim, width)

        blocks = []
        for _ in range(num_upsamples):
            blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
            blocks.append(nn.ConvTranspose1d(width, width, kernel_size=2, stride=2))
            blocks.append(nn.ReLU(True))
        blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Conv1d(width, coord_dim, kernel_size=1)
        self.use_final_interp = use_final_interp

    def forward(self, z):  # (B, M, L, D)
        B, M, L, D = z.shape
        assert D == self.embedding_dim, f"D mismatch: {D}!={self.embedding_dim}"
        assert L == self.num_joints * self.num_hands, f"L mismatch: {L}!={self.num_joints*self.num_hands}"

        z = z.view(B * M * L, D)                 # (BML, D)
        x = self.proj_in(z)                      # (BML, width)
        x = x.unsqueeze(-1)                      # (BML, width, 1)

        x = self.tcn(x)                          # (BML, width, T_dec)
        x = self.head(x)                         # (BML, C, T_dec)

        T_dec = x.shape[-1]
        if self.use_final_interp and T_dec != self.num_frames:
            x = F.interpolate(x, size=self.num_frames, mode="linear", align_corners=False)  # (BML, C, T)

        x = x.view(B, M, L, self.coord_dim, self.num_frames)                    # (B, M, L, C, T)
        x = x.view(B, M, self.num_joints, self.num_hands, self.coord_dim, self.num_frames)
        coords = x.permute(0, 1, 4, 5, 2, 3).contiguous()                       # (B, M, C, T, J, E)
        return coords