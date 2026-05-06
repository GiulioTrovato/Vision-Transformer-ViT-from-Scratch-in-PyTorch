import torch
from torch import nn

from src.modules import PatchEmbedding, TransformerBlock

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=256, img_size=32, num_heads = 8, n_classes=10, depth=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x) # (batch, 3, 32, 32) --> (batch, 64, 256)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        cls_out = x[:, 0] # batch and patches
        return self.classifier(cls_out)