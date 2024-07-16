import torch
import torch.nn as nn

class PostionalEncoding(nn.Module):
    def __init__(self, context_size, n_embed):
        super().__init__()
        self.context_size = context_size
        self.n_embed = n_embed
        # self.pos_projection = nn.Embedding(context_size, n_embed) 
        self.positions = nn.Parameter(torch.randn(context_size)) # ordering is also learnable

    def forward(self, x):
        x = x + self.positions
        # return self.pos_projection(x)
        # print("xshape after pos", x.shape)
        return x

class Model(nn.Module):
    def __init__(self, context_size, n_embed, n_hidden):
        super().__init__()
        self.context_size = context_size
        self.pos_encoder = PostionalEncoding(context_size, n_embed)
        self.ff = nn.Sequential(
            nn.Linear(context_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.ff(x)
        return x