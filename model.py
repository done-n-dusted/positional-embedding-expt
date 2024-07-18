import torch
import torch.nn as nn

class PostionalEncoding(nn.Module):
    def __init__(self, context_size):
        super().__init__()
        self.context_size = context_size
        self.positions = nn.Parameter(torch.randn(context_size)) # ordering is also learnable

    def forward(self, x):
        # [1, 2, 3, 4, 5]
        # [p1, p2, p3, p4, p5]
        out = x + self.positions
        return out

class Model(nn.Module):
    def __init__(self, context_size, n_hidden):
        super().__init__()
        self.context_size = context_size
        self.pos_encoder = PostionalEncoding(context_size)
        self.ff = nn.Sequential(
            nn.Linear(context_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.pos_encoder(x)
        out = self.ff(out)
        return out