import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()

        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, inputs):
        keys = self.w_key(inputs)
        queries = self.w_query(inputs)
        values = self.w_value(inputs)

        attn_scores = queries @ keys.transpose(-1, -2)
        
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale,
                                     dim=-1)

        context_vecs = attn_weights @ values
        return context_vecs
