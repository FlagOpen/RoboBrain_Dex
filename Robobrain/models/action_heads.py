import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnBlock(nn.Module):
    """
    x: (B, Q, D)  —— action query token embeddings
    ctx: (B, C, D) —— context tokens (visual + proprio token)
    """
    def __init__(self, dim=1536, nhead=16, ffn_dim=6144, dropout=0.1):
        super().__init__()
        # Self-Attn
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        # Cross-Attn (Q: x, K/V: ctx)
        self.norm2_x = nn.LayerNorm(dim)
        self.norm2_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        # FFN
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, ctx):
        # Self-attention on query tokens
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        # Cross-attention: queries attend to (visual + proprio) context
        x_norm = self.norm2_x(x)
        ctx_norm = self.norm2_ctx(ctx)
        x = x + self.cross_attn(x_norm, ctx_norm, ctx_norm, need_weights=False)[0]
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x
