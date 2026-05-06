import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=256, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Conv2d(in_channels, emb_size, patch_size, patch_size)

    def forward(self, x):
        x = self.project(x)
        # turn (1, 256, 8, 8) into (1, 64, 256)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class   MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        # dimensions
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        # stack Q, K, V matrices together --> (1, 256, 768)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # mix the results of different heads
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.qkv(x)
        # (1, 65, 768) -> (3, 8, 65, 32)
        qkv = rearrange(x, 'b n (qkv h d) -> (qkv b) h n d', h=self.num_heads, qkv=3)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        # k-transpose
        k = rearrange(k, 'b h n d -> b h d n')
        # self attention
        map = q @ k #heatmap (1, 8, 65, 65)
        map /= self.head_dim ** 0.5 # scaling
        map = F.softmax(map, dim=-1) # softmax
        z = map @ v
        z = rearrange(z, 'b h n d -> b n (h d)') # reassemble the heads (1, 65, 256)
        z = self.proj(z)
        return z


class MLP(nn.Module):
    def __init__(self, embed_dim=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads = 8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x
