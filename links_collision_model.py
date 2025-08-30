import torch.nn as nn


class LinksCollisionModel(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, device):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.norm = nn.LayerNorm(proj_dim)
        self.act = nn.GELU()
        from efficient_kan import KAN
        self.kan = KAN([proj_dim, 128, 64, 8], device=device)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return self.kan(x)
