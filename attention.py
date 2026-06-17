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

        context_length = keys.shape[-2]
        causal_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        causal_attn_scores = attn_scores.masked_fill(causal_mask.bool(), -torch.inf)
        
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(causal_attn_scores / scale,
                                     dim=-1)

        context_vecs = attn_weights @ values
        return context_vecs
