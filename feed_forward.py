import torch
import gelu
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        embedding_len = cfg["embedding_len"]
        self.layers = nn.Sequential(
                nn.Linear(embedding_len, 4*embedding_len),
                gelu.GELU(),
                nn.Linear(embedding_len*4, embedding_len)
            )

    def forward(self, x):
        return self.layers(x)
